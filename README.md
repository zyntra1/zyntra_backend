# Zyntra Backend API

A FastAPI-based backend application with authentication and role-based access control for Admins and Users/Employees.

## Features

- ✅ **Authentication System**
  - JWT token-based authentication
  - Separate login/register for Admins and Users
  - OAuth2 compatible endpoints

- ✅ **Role-Based Access Control**
  - Admin accounts with user management capabilities
  - Super Admin privileges
  - Regular Users/Employees under Admins

- ✅ **User Management**
  - Admins can create, read, update, and delete their employees
  - Users can update their own profiles
  - Hierarchical relationship between Admins and Users

- ✅ **Security**
  - Password hashing with bcrypt
  - JWT token authentication
  - Environment-based configuration
  - CORS support

## Project Structure

```
Zyntra_backend/
├── app/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py          # Configuration and settings
│   │   └── database.py        # Database connection
│   ├── models/
│   │   ├── __init__.py
│   │   └── user.py            # Admin and User models
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── auth.py            # Authentication endpoints
│   │   ├── admins.py          # Admin management endpoints
│   │   └── users.py           # User management endpoints
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── auth.py            # Pydantic schemas
│   └── utils/
│       ├── __init__.py
│       ├── auth.py            # Password hashing, JWT tokens
│       └── dependencies.py    # Auth dependencies
├── main.py                     # FastAPI application entry point
├── .env                        # Environment variables (create from .env.example)
├── .env.example               # Example environment configuration
└── .gitignore

```

## Setup Instructions

### 1. Clone and Navigate to Project

```bash
cd "/Users/adhivp/Desktop/Projects not deployed/Zyntra_backend"
```

### 2. Activate Virtual Environment

```bash
source zyntra_venv/bin/activate
```

### 3. Install Dependencies

The project requires the following packages. Install them using pip:

```bash
# Core FastAPI and server
pip install fastapi uvicorn[standard]

# Database
pip install sqlalchemy psycopg2-binary

# Authentication
pip install python-jose[cryptography] passlib[bcrypt] python-multipart

# Configuration
pip install pydantic pydantic-settings python-dotenv

# Email validation
pip install email-validator
```

After installation, freeze the requirements:

```bash
pip freeze > requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Edit `.env` and update the values:

```env
# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/zyntra_db

# JWT Configuration
SECRET_KEY=your-super-secret-key-change-this
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Application Configuration
APP_NAME=Zyntra API
APP_VERSION=1.0.0
DEBUG=True

# CORS Configuration
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000

# Admin Configuration
SUPER_ADMIN_EMAIL=admin@zyntra.com
SUPER_ADMIN_PASSWORD=changeme123
```

### 5. Set Up Database

Make sure your PostgreSQL server is running and create the database:

```sql
CREATE DATABASE zyntra_db;
```

The application will automatically create the tables on first run.

### 6. Run the Application

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload
```

The API will be available at:
- **API**: http://localhost:8000
- **Swagger Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Authentication

- `POST /api/auth/admin/register` - Register new admin
- `POST /api/auth/user/register` - Register new user/employee
- `POST /api/auth/login` - Login (admin or user)
- `POST /api/auth/token` - OAuth2 token endpoint

### Admin Management

- `GET /api/admins/me` - Get current admin info
- `GET /api/admins/me/employees` - Get admin's employees
- `PUT /api/admins/me` - Update current admin
- `GET /api/admins/` - List all admins (super admin only)
- `GET /api/admins/{admin_id}` - Get admin by ID (super admin only)
- `DELETE /api/admins/{admin_id}` - Delete admin (super admin only)

### User Management

- `GET /api/users/me` - Get current user info
- `PUT /api/users/me/update` - Update current user
- `GET /api/users/` - List all users under admin
- `POST /api/users/` - Create new user under admin
- `GET /api/users/{user_id}` - Get user by ID
- `PUT /api/users/{user_id}` - Update user
- `DELETE /api/users/{user_id}` - Delete user

## Usage Examples

### Register Admin

```bash
curl -X POST "http://localhost:8000/api/auth/admin/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@example.com",
    "username": "admin1",
    "password": "secure123",
    "full_name": "Admin User",
    "is_super_admin": false
  }'
```

### Login

```bash
curl -X POST "http://localhost:8000/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@example.com",
    "password": "secure123"
  }'
```

### Create User (as Admin)

```bash
curl -X POST "http://localhost:8000/api/users/" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "username": "user1",
    "password": "secure123",
    "full_name": "Employee User"
  }'
```

## Database Schema

### Admin Table
- `id`: Primary key
- `email`: Unique email address
- `username`: Unique username
- `hashed_password`: Bcrypt hashed password
- `full_name`: Full name
- `is_active`: Account status
- `is_super_admin`: Super admin flag
- `created_at`: Creation timestamp
- `updated_at`: Last update timestamp

### User Table
- `id`: Primary key
- `email`: Unique email address
- `username`: Unique username
- `hashed_password`: Bcrypt hashed password
- `full_name`: Full name
- `is_active`: Account status
- `admin_id`: Foreign key to Admin
- `created_at`: Creation timestamp
- `updated_at`: Last update timestamp

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black .
```

### Linting

```bash
flake8 .
```

## Security Notes

1. **Change the SECRET_KEY** in production to a strong random value
2. **Use HTTPS** in production
3. **Set DEBUG=False** in production
4. **Use strong passwords** for database and admin accounts
5. **Configure CORS** properly for your frontend domain
6. **Keep dependencies updated** regularly

## Database Support

The application supports multiple databases. Update `DATABASE_URL` in `.env`:

### PostgreSQL (Recommended)
```
postgresql://user:password@localhost:5432/dbname
```

### MySQL
```
mysql+pymysql://user:password@localhost:3306/dbname
```

### SQLite (Development Only)
```
sqlite:///./zyntra.db
```

## License

MIT License

## Support

For issues and questions, please open an issue on GitHub or contact the development team.
