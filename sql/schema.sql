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
    dividend_type VARCHAR NOT NULL,
    currency VARCHAR DEFAULT 'USD',
    source VARCHAR
);

CREATE TABLE IF NOT EXISTS events (
    event_id VARCHAR PRIMARY KEY,
    event_type VARCHAR NOT NULL,
    event_timestamp TIMESTAMPTZ NOT NULL,
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
