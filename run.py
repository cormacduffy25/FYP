from app import create_app

# Create an instance of the Flask application
app = create_app()

if __name__ == '__main__':
    # Run the app. In debug mode, the server will reload itself on code changes
    app.run(debug=True)
    