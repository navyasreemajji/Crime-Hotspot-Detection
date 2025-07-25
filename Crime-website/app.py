from flask import Flask, render_template, request, redirect, url_for, jsonify
import sqlite3

app = Flask(__name__)

# Initialize DB
def init_db():
    conn = sqlite3.connect('crime_posts.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            description TEXT
        )
    ''')
    conn.commit()
    conn.close()

@app.route('/')
def home():
    conn = sqlite3.connect('crime_posts.db')
    c = conn.cursor()
    c.execute('SELECT * FROM posts')
    posts = c.fetchall()
    conn.close()
    return render_template('report.html', posts=posts)

@app.route('/post', methods=['POST'])
def post():
    username = request.form['username']
    description = request.form['description']
    
    conn = sqlite3.connect('crime_posts.db')
    c = conn.cursor()
    c.execute('INSERT INTO posts (username, description) VALUES (?, ?)', (username, description))
    conn.commit()
    conn.close()
    
    return redirect(url_for('home'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
