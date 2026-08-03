from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import polars as pl

from vol_platform.pricing.greeks import black_scholes_greeks
from vol_platform.pricing.implied_vol import solve_implied_volatility
from vol_platform.strategy.config import StrategyConfig
from vol_platform.types import OptionType, PricingModel


@dataclass(frozen=True, slots=True)
class BacktestFrames:
    trades: pl.DataFrame
    attribution: pl.DataFrame
    hedge_log: pl.DataFrame
    rejections: pl.DataFrame


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_date(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _option_type(value: object) -> OptionType | None:
    text = str(value or "").strip().lower()
    if text in {"call", "c"}:
        return OptionType.CALL
    if text in {"put", "p"}:
        return OptionType.PUT
    return None


def _underlying_symbol(row: dict[str, Any]) -> str:
    return str(row.get("underlying_symbol") or row.get("root") or "").strip().upper()


def _contract_id(row: dict[str, Any]) -> str:
    explicit = str(row.get("symbol") or row.get("contract_symbol") or "").strip()
    if explicit:
        return explicit
    expiration = _as_date(row.get("expiration"))
    return "|".join(
        [
            _underlying_symbol(row),
            expiration.isoformat() if expiration else "",
            str(row.get("strike")),
            str(row.get("option_type")),
        ]
    )


def _mid(row: dict[str, Any]) -> float | None:
    bid = _finite(row.get("bid"))
    ask = _finite(row.get("ask"))
    if bid is None or ask is None or bid < 0.0 or ask <= 0.0 or ask < bid:
        return None
    return 0.5 * (bid + ask)


def _spread(row: dict[str, Any]) -> float | None:
    bid = _finite(row.get("bid"))
    ask = _finite(row.get("ask"))
    if bid is None or ask is None or ask < bid:
        return None
    return ask - bid


def _relative_spread(row: dict[str, Any]) -> float | None:
    midpoint = _mid(row)
    spread = _spread(row)
    if midpoint is None or spread is None or midpoint <= 0.0:
        return None
    return spread / midpoint


def _price_map(underlying: pl.DataFrame) -> dict[tuple[str, date], float]:
    output: dict[tuple[str, date], float] = {}
    for row in underlying.sort("timestamp").iter_rows(named=True):
        current_date = _as_date(row.get("timestamp"))
        price = _finite(row.get("underlying_price", row.get("last")))
        symbol = str(row.get("symbol") or "").strip().upper()
        if current_date is not None and price is not None and symbol:
            output[(symbol, current_date)] = price
    return output


def _quote_indexes(
    option_quotes: pl.DataFrame,
) -> tuple[
    dict[tuple[str, date], list[dict[str, Any]]],
    dict[tuple[str, date], dict[str, Any]],
]:
    by_symbol_date: dict[tuple[str, date], list[dict[str, Any]]] = {}
    by_contract_date: dict[tuple[str, date], dict[str, Any]] = {}
    for row in option_quotes.sort("quote_timestamp").iter_rows(named=True):
        current_date = _as_date(row.get("quote_timestamp"))
        symbol = _underlying_symbol(row)
        option_type = _option_type(row.get("option_type"))
        expiration = _as_date(row.get("expiration"))
        if current_date is None or not symbol or option_type is None or expiration is None:
            continue
        normalized = dict(row)
        normalized["option_type"] = option_type.value
        normalized["expiration"] = expiration
        normalized["quote_date"] = current_date
        normalized["contract_id"] = _contract_id(row)
        by_symbol_date.setdefault((symbol, current_date), []).append(normalized)
        by_contract_date[(normalized["contract_id"], current_date)] = normalized
    return by_symbol_date, by_contract_date


def _event_symbol(signal: dict[str, Any]) -> str:
    explicit = str(signal.get("symbol") or "").strip().upper()
    if explicit:
        return explicit
    raw = str(signal.get("symbols") or "").replace(";", ",")
    return next((item.strip().upper() for item in raw.split(",") if item.strip()), "")


def _select_pair(
    entry_rows: list[dict[str, Any]],
    *,
    entry_date: date,
    spot: float,
    config: StrategyConfig,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    pairs: dict[tuple[date, float], dict[str, dict[str, Any]]] = {}
    for row in entry_rows:
        expiration = _as_date(row.get("expiration"))
        strike = _finite(row.get("strike"))
        option_type = _option_type(row.get("option_type"))
        midpoint = _mid(row)
        relative_spread = _relative_spread(row)
        volume = int(_finite(row.get("volume")) or 0)
        open_interest = int(_finite(row.get("open_interest")) or 0)
        if (
            expiration is None
            or strike is None
            or strike <= 0.0
            or option_type is None
            or midpoint is None
            or midpoint <= 0.0
            or relative_spread is None
            or relative_spread > config.maximum_relative_spread
            or volume < config.minimum_volume
            or open_interest < config.minimum_open_interest
        ):
            continue
        dte = (expiration - entry_date).days
        if not config.minimum_dte_days <= dte <= config.maximum_dte_days:
            continue
        pairs.setdefault((expiration, strike), {})[option_type.value] = row

    candidates: list[tuple[tuple[float, float, float], dict[str, Any], dict[str, Any]]] = []
    for (expiration, strike), legs in pairs.items():
        call = legs.get(OptionType.CALL.value)
        put = legs.get(OptionType.PUT.value)
        if call is None or put is None:
            continue
        dte_distance = abs((expiration - entry_date).days - config.target_dte_days)
        moneyness_distance = abs(math.log(strike / spot))
        spread_score = (_relative_spread(call) or 0.0) + (_relative_spread(put) or 0.0)
        candidates.append(((dte_distance, moneyness_distance, spread_score), call, put))
    if not candidates:
        return None
    _, call, put = min(candidates, key=lambda item: item[0])
    return call, put


def _implied_volatility(
    row: dict[str, Any],
    *,
    spot: float,
    current_date: date,
    rate: float,
    dividend_yield: float,
) -> float | None:
    direct = _finite(row.get("implied_volatility", row.get("mid_implied_volatility")))
    if direct is not None and direct > 0.0:
        return direct
    expiration = _as_date(row.get("expiration"))
    strike = _finite(row.get("strike"))
    option_type = _option_type(row.get("option_type"))
    midpoint = _mid(row)
    if expiration is None or strike is None or option_type is None or midpoint is None:
        return None
    time_to_expiry = max((expiration - current_date).days / 365.0, 1.0 / 365.0)
    result = solve_implied_volatility(
        midpoint,
        spot,
        strike,
        time_to_expiry,
        rate,
        option_type,
        PricingModel.BLACK_SCHOLES,
        dividend_yield,
    )
    return result.volatility if result.converged else None


def _leg_greeks(
    row: dict[str, Any],
    *,
    spot: float,
    current_date: date,
    rate: float,
    dividend_yield: float,
) -> tuple[float, float, float, float, float] | None:
    expiration = _as_date(row.get("expiration"))
    strike = _finite(row.get("strike"))
    option_type = _option_type(row.get("option_type"))
    volatility = _implied_volatility(
        row,
        spot=spot,
        current_date=current_date,
        rate=rate,
        dividend_yield=dividend_yield,
    )
    if expiration is None or strike is None or option_type is None or volatility is None:
        return None
    time_to_expiry = max((expiration - current_date).days / 365.0, 1.0 / 365.0)
    greeks = black_scholes_greeks(
        spot,
        strike,
        time_to_expiry,
        rate,
        volatility,
        option_type,
        dividend_yield,
    )
    return greeks.delta, greeks.gamma, greeks.vega, greeks.theta, volatility


def _direction(signal: dict[str, Any], config: StrategyConfig) -> int:
    prediction = _finite(signal.get("linear_prediction", signal.get("prediction")))
    if prediction is None:
        return 0
    if prediction > config.prediction_threshold and config.allow_short:
        return -1
    if prediction < -config.prediction_threshold and config.allow_long:
        return 1
    return 0


def _execution_price(row: dict[str, Any], *, buying: bool, slippage: float) -> float:
    bid = float(row["bid"])
    ask = float(row["ask"])
    spread = ask - bid
    return ask + slippage * spread if buying else max(bid - slippage * spread, 0.0)


def _trade_dates(
    reaction_date: date,
    price_dates: list[date],
    config: StrategyConfig,
) -> tuple[date, date] | None:
    available = sorted(set(price_dates))
    reaction_index = next(
        (index for index, item in enumerate(available) if item >= reaction_date),
        None,
    )
    if reaction_index is None:
        return None
    entry_index = reaction_index - config.entry_days_before_event
    exit_index = reaction_index + config.holding_period_days
    if entry_index < 0 or exit_index >= len(available) or exit_index <= entry_index:
        return None
    return available[entry_index], available[exit_index]


def run_contract_backtest(
    signals: pl.DataFrame,
    option_quotes: pl.DataFrame,
    underlying: pl.DataFrame,
    config: StrategyConfig,
) -> BacktestFrames:
    """Run an event-conditioned, delta-hedged straddle backtest.

    Positive model predictions short the straddle because Week 5 defines the
    target as expected move minus realized move. Negative predictions buy it.
    """

    config.validate()
    prices = _price_map(underlying)
    price_dates_by_symbol: dict[str, list[date]] = {}
    for symbol, current_date in prices:
        price_dates_by_symbol.setdefault(symbol, []).append(current_date)
    quotes_by_symbol_date, quotes_by_contract_date = _quote_indexes(option_quotes)

    trades: list[dict[str, Any]] = []
    attributions: list[dict[str, Any]] = []
    hedge_rows: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    latest_exit_by_symbol: dict[str, date] = {}
    active_capital: list[tuple[date, float]] = []

    ordered_signals = signals.sort("event_timestamp")
    for signal in ordered_signals.iter_rows(named=True):
        event_id = str(signal.get("event_id") or "")
        symbol = _event_symbol(signal)
        event_type = str(signal.get("event_type") or "other").lower()
        event_timestamp = signal.get("event_timestamp")
        reaction_date = _as_date(signal.get("reaction_date", event_timestamp))

        def reject(
            reason: str,
            event_id: str = event_id,
            symbol: str = symbol,
            event_type: str = event_type,
            event_timestamp: Any = event_timestamp,
        ) -> None:
            rejections.append(
                {
                    "event_id": event_id,
                    "symbol": symbol,
                    "event_type": event_type,
                    "event_timestamp": event_timestamp,
                    "reason": reason,
                }
            )

        if not event_id or not symbol or reaction_date is None:
            reject("missing_event_identifier_symbol_or_date")
            continue
        if event_type not in config.tradable_event_types:
            reject("event_type_not_tradable")
            continue
        direction = _direction(signal, config)
        if direction == 0:
            reject("signal_below_threshold")
            continue
        dates = _trade_dates(
            reaction_date,
            price_dates_by_symbol.get(symbol, []),
            config,
        )
        if dates is None:
            reject("incomplete_entry_or_exit_window")
            continue
        entry_date, exit_date = dates
        prior_exit = latest_exit_by_symbol.get(symbol)
        if prior_exit is not None and entry_date <= prior_exit:
            reject("overlapping_symbol_position")
            continue
        entry_spot = prices.get((symbol, entry_date))
        exit_spot = prices.get((symbol, exit_date))
        if entry_spot is None or exit_spot is None:
            reject("missing_underlying_price")
            continue
        pair = _select_pair(
            quotes_by_symbol_date.get((symbol, entry_date), []),
            entry_date=entry_date,
            spot=entry_spot,
            config=config,
        )
        if pair is None:
            reject("no_liquid_atm_call_put_pair")
            continue
        entry_call, entry_put = pair
        call_id = str(entry_call["contract_id"])
        put_id = str(entry_put["contract_id"])
        exit_call = quotes_by_contract_date.get((call_id, exit_date))
        exit_put = quotes_by_contract_date.get((put_id, exit_date))
        if (
            exit_call is None
            or exit_put is None
            or _mid(exit_call) is None
            or _mid(exit_put) is None
        ):
            reject("missing_exit_contract_quotes")
            continue

        call_multiplier = int(_finite(entry_call.get("multiplier")) or config.contract_multiplier)
        put_multiplier = int(_finite(entry_put.get("multiplier")) or config.contract_multiplier)
        if call_multiplier != put_multiplier:
            reject("call_put_multiplier_mismatch")
            continue
        multiplier = call_multiplier
        entry_straddle_mid = float(_mid(entry_call) or 0.0) + float(_mid(entry_put) or 0.0)
        per_contract_capital = (
            entry_straddle_mid * multiplier
            if direction > 0
            else max(
                entry_straddle_mid * multiplier,
                entry_spot * multiplier * config.short_margin_fraction,
            )
        )
        per_trade_limit = config.portfolio_capital * config.maximum_capital_fraction_per_trade
        per_trade_contracts = int(per_trade_limit // max(per_contract_capital, 1.0e-12))
        active_capital = [
            (position_exit, capital)
            for position_exit, capital in active_capital
            if position_exit >= entry_date
        ]
        available_portfolio_capital = max(
            config.portfolio_capital - sum(capital for _, capital in active_capital),
            0.0,
        )
        portfolio_contracts = int(available_portfolio_capital // max(per_contract_capital, 1.0e-12))
        contracts = min(
            config.contracts_per_trade,
            config.maximum_contracts,
            per_trade_contracts,
            portfolio_contracts,
        )
        if contracts < 1:
            reason = (
                "capital_limit_prevents_trade"
                if per_trade_contracts < 1
                else "portfolio_capital_limit_prevents_trade"
            )
            reject(reason)
            continue
        quantity = direction * contracts

        entry_call_mid = float(_mid(entry_call) or 0.0)
        entry_put_mid = float(_mid(entry_put) or 0.0)
        exit_call_mid = float(_mid(exit_call) or 0.0)
        exit_put_mid = float(_mid(exit_put) or 0.0)
        option_pnl = (
            quantity * multiplier * (exit_call_mid + exit_put_mid - entry_call_mid - entry_put_mid)
        )

        entry_execution = 0.0
        exit_execution = 0.0
        for entry_leg, exit_leg in ((entry_call, exit_call), (entry_put, exit_put)):
            if quantity > 0:
                entry_execution -= (
                    quantity
                    * multiplier
                    * _execution_price(
                        entry_leg,
                        buying=True,
                        slippage=config.option_slippage_fraction_of_spread,
                    )
                )
                exit_execution += (
                    quantity
                    * multiplier
                    * _execution_price(
                        exit_leg,
                        buying=False,
                        slippage=config.option_slippage_fraction_of_spread,
                    )
                )
            else:
                entry_execution -= (
                    quantity
                    * multiplier
                    * _execution_price(
                        entry_leg,
                        buying=False,
                        slippage=config.option_slippage_fraction_of_spread,
                    )
                )
                exit_execution += (
                    quantity
                    * multiplier
                    * _execution_price(
                        exit_leg,
                        buying=True,
                        slippage=config.option_slippage_fraction_of_spread,
                    )
                )
        executable_option_pnl = entry_execution + exit_execution
        option_execution_cost = max(option_pnl - executable_option_pnl, 0.0)
        entry_option_commissions = 2.0 * contracts * config.commission_per_contract
        exit_option_commissions = 2.0 * contracts * config.commission_per_contract
        option_commissions = entry_option_commissions + exit_option_commissions

        window_dates = [
            current
            for current in price_dates_by_symbol.get(symbol, [])
            if entry_date <= current <= exit_date
        ]
        window_dates = sorted(set(window_dates))
        if len(window_dates) < 2:
            reject("insufficient_hedge_dates")
            continue
        if any(
            (call_id, current) not in quotes_by_contract_date
            or (put_id, current) not in quotes_by_contract_date
            for current in window_dates
        ):
            reject("missing_intermediate_contract_quotes")
            continue

        entry_call_greeks = _leg_greeks(
            entry_call,
            spot=entry_spot,
            current_date=entry_date,
            rate=config.financing_rate,
            dividend_yield=config.dividend_yield,
        )
        entry_put_greeks = _leg_greeks(
            entry_put,
            spot=entry_spot,
            current_date=entry_date,
            rate=config.financing_rate,
            dividend_yield=config.dividend_yield,
        )
        if entry_call_greeks is None or entry_put_greeks is None:
            reject("unable_to_calculate_entry_greeks")
            continue

        option_delta = quantity * multiplier * (entry_call_greeks[0] + entry_put_greeks[0])
        hedge_shares = -option_delta
        hedge_cost = abs(hedge_shares) * entry_spot * config.hedge_cost_bps / 10_000.0
        hedge_turnover = abs(hedge_shares) * entry_spot
        cash_balance = entry_execution - hedge_shares * entry_spot
        cash_balance -= entry_option_commissions + hedge_cost
        hedge_pnl = 0.0
        financing_pnl = 0.0
        delta_attribution = 0.0
        gamma_attribution = 0.0
        vega_attribution = 0.0
        theta_attribution = 0.0

        hedge_rows.append(
            {
                "event_id": event_id,
                "symbol": symbol,
                "date": entry_date,
                "action": "initial_hedge",
                "spot": entry_spot,
                "option_delta": option_delta,
                "target_hedge_shares": hedge_shares,
                "hedge_trade_shares": hedge_shares,
                "hedge_cost": hedge_cost,
            }
        )

        for index in range(len(window_dates) - 1):
            current_date = window_dates[index]
            next_date = window_dates[index + 1]
            current_spot = prices[(symbol, current_date)]
            next_spot = prices[(symbol, next_date)]
            dt = max((next_date - current_date).days / 365.0, 0.0)
            d_spot = next_spot - current_spot
            hedge_pnl += hedge_shares * d_spot
            interest = cash_balance * (math.exp(config.financing_rate * dt) - 1.0)
            financing_pnl += interest
            cash_balance += interest

            current_call = quotes_by_contract_date.get((call_id, current_date))
            current_put = quotes_by_contract_date.get((put_id, current_date))
            next_call = quotes_by_contract_date.get((call_id, next_date))
            next_put = quotes_by_contract_date.get((put_id, next_date))
            if current_call and current_put and next_call and next_put:
                call_greeks = _leg_greeks(
                    current_call,
                    spot=current_spot,
                    current_date=current_date,
                    rate=config.financing_rate,
                    dividend_yield=config.dividend_yield,
                )
                put_greeks = _leg_greeks(
                    current_put,
                    spot=current_spot,
                    current_date=current_date,
                    rate=config.financing_rate,
                    dividend_yield=config.dividend_yield,
                )
                if call_greeks and put_greeks:
                    delta_attribution += (
                        quantity * multiplier * (call_greeks[0] + put_greeks[0]) * d_spot
                    )
                    gamma_attribution += (
                        0.5 * quantity * multiplier * (call_greeks[1] + put_greeks[1]) * d_spot**2
                    )
                    next_call_vol = _implied_volatility(
                        next_call,
                        spot=next_spot,
                        current_date=next_date,
                        rate=config.financing_rate,
                        dividend_yield=config.dividend_yield,
                    )
                    next_put_vol = _implied_volatility(
                        next_put,
                        spot=next_spot,
                        current_date=next_date,
                        rate=config.financing_rate,
                        dividend_yield=config.dividend_yield,
                    )
                    if next_call_vol is not None:
                        vega_attribution += (
                            quantity
                            * multiplier
                            * call_greeks[2]
                            * (next_call_vol - call_greeks[4])
                        )
                    if next_put_vol is not None:
                        vega_attribution += (
                            quantity * multiplier * put_greeks[2] * (next_put_vol - put_greeks[4])
                        )
                    theta_attribution += (
                        quantity * multiplier * (call_greeks[3] + put_greeks[3]) * dt
                    )

            should_rehedge = (
                next_date != exit_date and (index + 1) % config.hedge_frequency_days == 0
            )
            if should_rehedge and next_call and next_put:
                next_call_greeks = _leg_greeks(
                    next_call,
                    spot=next_spot,
                    current_date=next_date,
                    rate=config.financing_rate,
                    dividend_yield=config.dividend_yield,
                )
                next_put_greeks = _leg_greeks(
                    next_put,
                    spot=next_spot,
                    current_date=next_date,
                    rate=config.financing_rate,
                    dividend_yield=config.dividend_yield,
                )
                if next_call_greeks and next_put_greeks:
                    next_option_delta = (
                        quantity * multiplier * (next_call_greeks[0] + next_put_greeks[0])
                    )
                    target_hedge = -next_option_delta
                    hedge_trade = target_hedge - hedge_shares
                    trade_cost = abs(hedge_trade) * next_spot * config.hedge_cost_bps / 10_000.0
                    hedge_cost += trade_cost
                    hedge_turnover += abs(hedge_trade) * next_spot
                    cash_balance -= hedge_trade * next_spot + trade_cost
                    hedge_shares = target_hedge
                    hedge_rows.append(
                        {
                            "event_id": event_id,
                            "symbol": symbol,
                            "date": next_date,
                            "action": "rebalance",
                            "spot": next_spot,
                            "option_delta": next_option_delta,
                            "target_hedge_shares": target_hedge,
                            "hedge_trade_shares": hedge_trade,
                            "hedge_cost": trade_cost,
                        }
                    )

        close_cost = abs(hedge_shares) * exit_spot * config.hedge_cost_bps / 10_000.0
        hedge_cost += close_cost
        hedge_turnover += abs(hedge_shares) * exit_spot
        hedge_rows.append(
            {
                "event_id": event_id,
                "symbol": symbol,
                "date": exit_date,
                "action": "close_hedge",
                "spot": exit_spot,
                "option_delta": None,
                "target_hedge_shares": 0.0,
                "hedge_trade_shares": -hedge_shares,
                "hedge_cost": close_cost,
            }
        )

        transaction_costs = option_execution_cost + option_commissions + hedge_cost
        gross_pnl = option_pnl + hedge_pnl + financing_pnl
        net_pnl = gross_pnl - transaction_costs
        midpoint_upper_bound_pnl = gross_pnl - option_commissions - hedge_cost
        capital_at_risk = per_contract_capital * contracts
        net_return = net_pnl / max(capital_at_risk, 1.0e-12)
        gross_return = gross_pnl / max(capital_at_risk, 1.0e-12)
        midpoint_upper_bound_return = midpoint_upper_bound_pnl / max(capital_at_risk, 1.0e-12)
        residual = option_pnl - (
            delta_attribution + gamma_attribution + vega_attribution + theta_attribution
        )
        option_turnover = (
            multiplier * contracts * (entry_call_mid + entry_put_mid + exit_call_mid + exit_put_mid)
        )
        total_turnover = option_turnover + hedge_turnover
        expiration = _as_date(entry_call.get("expiration"))
        event_year = reaction_date.year

        trade = dict(signal)
        trade.update(
            {
                "symbol": symbol,
                "event_type": event_type,
                "reaction_date": reaction_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "holding_period_days": config.holding_period_days,
                "hedge_frequency_days": config.hedge_frequency_days,
                "strategy_name": config.strategy_name,
                "strategy_position": "long_straddle" if direction > 0 else "short_straddle",
                "direction": direction,
                "contracts": contracts,
                "call_contract": call_id,
                "put_contract": put_id,
                "strike": _finite(entry_call.get("strike")),
                "expiration": expiration,
                "entry_dte_days": (expiration - entry_date).days if expiration else None,
                "entry_spot": entry_spot,
                "exit_spot": exit_spot,
                "entry_call_mid": entry_call_mid,
                "entry_put_mid": entry_put_mid,
                "exit_call_mid": exit_call_mid,
                "exit_put_mid": exit_put_mid,
                "capital_at_risk": capital_at_risk,
                "option_pnl": option_pnl,
                "executable_option_pnl": executable_option_pnl,
                "hedge_pnl": hedge_pnl,
                "transaction_costs": transaction_costs,
                "option_execution_cost": option_execution_cost,
                "option_commissions": option_commissions,
                "hedge_cost": hedge_cost,
                "financing_pnl": financing_pnl,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "midpoint_upper_bound_pnl": midpoint_upper_bound_pnl,
                "gross_return": gross_return,
                "net_return": net_return,
                "midpoint_upper_bound_return": midpoint_upper_bound_return,
                "turnover_notional": total_turnover,
                "event_year": event_year,
                "pnl_method": "exact_contract_level_bid_ask_delta_hedged",
            }
        )
        trades.append(trade)
        attributions.append(
            {
                "event_id": event_id,
                "symbol": symbol,
                "event_timestamp": signal.get("event_timestamp"),
                "entry_date": entry_date,
                "exit_date": exit_date,
                "delta_pnl": delta_attribution,
                "gamma_pnl": gamma_attribution,
                "vega_pnl": vega_attribution,
                "theta_pnl": theta_attribution,
                "option_residual_pnl": residual,
                "hedge_pnl": hedge_pnl,
                "transaction_cost_pnl": -transaction_costs,
                "financing_pnl": financing_pnl,
                "net_pnl": net_pnl,
            }
        )
        latest_exit_by_symbol[symbol] = exit_date
        active_capital.append((exit_date, capital_at_risk))

    trade_frame = (
        pl.DataFrame(trades).sort("event_timestamp")
        if trades
        else pl.DataFrame(
            schema={
                "event_id": pl.String,
                "event_timestamp": pl.Datetime(time_zone="UTC"),
                "symbol": pl.String,
                "net_return": pl.Float64,
                "net_pnl": pl.Float64,
            }
        )
    )
    attribution_frame = (
        pl.DataFrame(attributions).sort("event_timestamp")
        if attributions
        else pl.DataFrame(
            schema={
                "event_id": pl.String,
                "event_timestamp": pl.Datetime(time_zone="UTC"),
                "symbol": pl.String,
                "net_pnl": pl.Float64,
            }
        )
    )
    hedge_frame = (
        pl.DataFrame(hedge_rows).sort(["event_id", "date"])
        if hedge_rows
        else pl.DataFrame(
            schema={
                "event_id": pl.String,
                "symbol": pl.String,
                "date": pl.Date,
                "action": pl.String,
            }
        )
    )
    rejection_frame = (
        pl.DataFrame(rejections).sort("event_timestamp")
        if rejections
        else pl.DataFrame(
            schema={
                "event_id": pl.String,
                "symbol": pl.String,
                "event_type": pl.String,
                "event_timestamp": pl.Datetime(time_zone="UTC"),
                "reason": pl.String,
            }
        )
    )
    return BacktestFrames(
        trades=trade_frame,
        attribution=attribution_frame,
        hedge_log=hedge_frame,
        rejections=rejection_frame,
    )
