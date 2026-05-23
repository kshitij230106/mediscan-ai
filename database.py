import sqlite3

# Connect database
conn = sqlite3.connect("predictions.db")

# Create cursor
cursor = conn.cursor()

# Create table-
cursor.execute("""

CREATE TABLE IF NOT EXISTS prediction_history (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    disease_type TEXT,

    prediction TEXT,

    confidence REAL,

    risk_level TEXT,

    health_score INTEGER,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)

""")

# Save changes
conn.commit()

# Close connection
conn.close()

print("Database created successfully!")