# Multi-stage build for AIDB server deployment
FROM rust:1.83-slim as builder

# Install required dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Cargo files
COPY Cargo.toml Cargo.lock ./
COPY crates/ ./crates/

# Build release binary
RUN cargo build --release --package aidb-server

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -r -s /bin/false -m -d /app aidb

# Set working directory
WORKDIR /app

# Copy binary from builder stage
COPY --from=builder /app/target/release/aidb-server /usr/local/bin/aidb-server

# Change ownership
RUN chown -R aidb:aidb /app

# Switch to app user
USER aidb

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Set environment variables
ENV RUST_LOG=info
ENV AIDB_HOST=0.0.0.0
ENV AIDB_PORT=8080

# Run the server
CMD ["aidb-server"]