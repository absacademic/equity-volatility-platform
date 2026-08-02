CREATE TABLE IF NOT EXISTS option_quotes (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR NOT NULL,
    underlying_symbol VARCHAR NOT NULL,
    expiration DATE NOT NULL,
    strike DOUBLE NOT NULL CHECK (strike > 0),
    option_type VARCHAR NOT NULL CHECK (option_type IN ('call', 'put')),
    bid DOUBLE NOT NULL CHECK (bid >= 0),
    ask DOUBLE NOT NULL CHECK (ask >= bid),
    last DOUBLE,
    bid_size BIGINT,
    ask_size BIGINT,
    volume BIGINT,
    open_interest BIGINT,
    exchange VARCHAR,
    currency VARCHAR DEFAULT 'USD',
    multiplier INTEGER DEFAULT 100
);

CREATE TABLE IF NOT EXISTS underlying_prices (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR NOT NULL,
    bid DOUBLE,
    ask DOUBLE,
    last DOUBLE,
    volume BIGINT,
    currency VARCHAR DEFAULT 'USD'
);

CREATE TABLE IF NOT EXISTS rate_curve (
    as_of_date DATE NOT NULL,
    maturity_date DATE NOT NULL,
    rate DOUBLE NOT NULL,
    currency VARCHAR DEFAULT 'USD',
    source VARCHAR NOT NULL,
    PRIMARY KEY (as_of_date, maturity_date, currency, source)
);

CREATE TABLE IF NOT EXISTS dividends (
    symbol VARCHAR NOT NULL,
    ex_date DATE NOT NULL,
    amount DOUBLE NOT NULL CHECK (amount >= 0),
    payment_date DATE,
    known_timestamp TIMESTAMPTZ,
    dividend_type VARCHAR NOT NULL,
    currency VARCHAR DEFAULT 'USD',
    source VARCHAR
);

CREATE TABLE IF NOT EXISTS events (
    event_id VARCHAR PRIMARY KEY,
    event_type VARCHAR NOT NULL,
    event_timestamp TIMESTAMPTZ NOT NULL,
    known_timestamp TIMESTAMPTZ,
    title VARCHAR NOT NULL,
    symbols VARCHAR[],
    source VARCHAR,
    expected BOOLEAN DEFAULT TRUE,
    metadata JSON
);

CREATE TABLE IF NOT EXISTS implied_volatilities (
    quote_timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR NOT NULL,
    expiration DATE NOT NULL,
    strike DOUBLE NOT NULL CHECK (strike > 0),
    option_type VARCHAR NOT NULL CHECK (option_type IN ('call', 'put')),
    model VARCHAR NOT NULL CHECK (model IN ('black_scholes', 'black_76')),
    option_price DOUBLE NOT NULL CHECK (option_price >= 0),
    spot DOUBLE,
    forward DOUBLE,
    rate DOUBLE NOT NULL,
    dividend_yield DOUBLE DEFAULT 0,
    implied_volatility DOUBLE,
    status VARCHAR NOT NULL,
    method VARCHAR NOT NULL,
    iterations INTEGER DEFAULT 0,
    residual DOUBLE
);

CREATE TABLE IF NOT EXISTS forward_estimates (
    expiration DATE NOT NULL,
    time_to_expiry DOUBLE NOT NULL,
    interpolated_rate DOUBLE NOT NULL,
    discount_factor DOUBLE NOT NULL,
    forward DOUBLE,
    pair_count INTEGER NOT NULL,
    forward_std DOUBLE,
    relative_dispersion DOUBLE,
    forward_range DOUBLE,
    average_pair_quality DOUBLE,
    reliability VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS smile_model_comparison (
    model VARCHAR NOT NULL,
    weighting VARCHAR NOT NULL,
    expiration_count INTEGER NOT NULL,
    failed_fit_rate DOUBLE NOT NULL,
    average_rmse DOUBLE,
    maximum_residual DOUBLE,
    average_coverage DOUBLE NOT NULL,
    average_stability DOUBLE NOT NULL
);

CREATE TABLE IF NOT EXISTS arbitrage_diagnostics (
    quote_date DATE NOT NULL,
    symbol VARCHAR NOT NULL,
    expiration DATE NOT NULL,
    source VARCHAR NOT NULL,
    model VARCHAR,
    weighting VARCHAR,
    "check" VARCHAR NOT NULL,
    option_type VARCHAR,
    location DOUBLE,
    value DOUBLE,
    tolerance DOUBLE NOT NULL,
    is_violation BOOLEAN NOT NULL,
    severity VARCHAR NOT NULL,
    resolved BOOLEAN NOT NULL,
    message VARCHAR
);

CREATE TABLE IF NOT EXISTS surface_adjustments (
    quote_date DATE NOT NULL,
    symbol VARCHAR NOT NULL,
    expiration DATE NOT NULL,
    model VARCHAR NOT NULL,
    weighting VARCHAR NOT NULL,
    action VARCHAR NOT NULL,
    "check" VARCHAR NOT NULL,
    before_value DOUBLE,
    after_value DOUBLE,
    reason VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS standardized_delta_points (
    quote_date DATE NOT NULL,
    symbol VARCHAR NOT NULL,
    expiration DATE NOT NULL,
    time_to_expiry DOUBLE NOT NULL,
    point VARCHAR NOT NULL,
    option_type VARCHAR,
    target_delta DOUBLE,
    actual_delta DOUBLE,
    delta_error DOUBLE,
    delta_convention VARCHAR NOT NULL,
    strike DOUBLE,
    forward_moneyness DOUBLE,
    implied_volatility DOUBLE,
    total_variance DOUBLE,
    model VARCHAR,
    weighting VARCHAR,
    status VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_volatility_features (
    quote_date DATE NOT NULL,
    quote_timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR NOT NULL,
    expiration DATE NOT NULL,
    time_to_expiry DOUBLE NOT NULL,
    forward DOUBLE,
    atm_implied_volatility DOUBLE,
    downside_skew_25 DOUBLE,
    risk_reversal_25 DOUBLE,
    butterfly_25 DOUBLE,
    wing_curvature_10_25 DOUBLE,
    atm_term_structure_slope DOUBLE,
    skew_term_structure_slope DOUBLE,
    iv_bid_ask_width DOUBLE,
    surface_residual_rmse DOUBLE,
    total_option_volume BIGINT,
    total_open_interest BIGINT,
    realized_volatility_20d DOUBLE,
    vrp_volatility_20d DOUBLE,
    vrp_variance_20d DOUBLE,
    event_count_to_expiry INTEGER,
    material_arbitrage_violation_count INTEGER NOT NULL,
    standardized_points_complete BOOLEAN NOT NULL,
    chain_valid BOOLEAN NOT NULL,
    PRIMARY KEY (quote_date, symbol, expiration)
);

CREATE TABLE IF NOT EXISTS event_study_events (
    event_id VARCHAR PRIMARY KEY,
    event_type VARCHAR NOT NULL,
    event_timestamp TIMESTAMPTZ NOT NULL,
    known_timestamp TIMESTAMPTZ,
    market_session VARCHAR NOT NULL,
    event_date DATE NOT NULL,
    reaction_date DATE NOT NULL,
    title VARCHAR,
    symbols VARCHAR,
    source VARCHAR,
    expected BOOLEAN NOT NULL,
    point_in_time_valid BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS event_study_observations (
    event_id VARCHAR PRIMARY KEY,
    event_timestamp TIMESTAMPTZ NOT NULL,
    period VARCHAR NOT NULL,
    symbol VARCHAR NOT NULL,
    atm_volatility DOUBLE,
    skew DOUBLE,
    term_structure DOUBLE,
    expected_move DOUBLE,
    volume_change DOUBLE,
    open_interest_change DOUBLE,
    iv_percentile DOUBLE,
    surface_dislocation DOUBLE,
    signed_return DOUBLE,
    absolute_return DOUBLE,
    expected_minus_realized_move DOUBLE,
    market_overestimated BOOLEAN,
    post_event_iv_collapse DOUBLE,
    atm_volatility_change DOUBLE,
    skew_change DOUBLE,
    long_straddle_gross_return DOUBLE,
    estimated_transaction_cost DOUBLE,
    long_straddle_net_return DOUBLE,
    delta_hedged_straddle_return DOUBLE
);

CREATE TABLE IF NOT EXISTS event_strategy_results (
    event_id VARCHAR PRIMARY KEY,
    event_timestamp TIMESTAMPTZ NOT NULL,
    period VARCHAR NOT NULL,
    strategy_position DOUBLE NOT NULL,
    strategy_gross_return DOUBLE NOT NULL,
    estimated_transaction_cost DOUBLE NOT NULL,
    strategy_net_return DOUBLE NOT NULL,
    strategy_cumulative_return DOUBLE NOT NULL
);
