"""
Database initialization script.
Run this to create all database tables.
"""

from app.core.database import engine, Base
from app.models import Admin, User
from app.utils import get_password_hash
from app.core.config import settings
from sqlalchemy.orm import Session


def init_db():
    """Initialize database tables."""
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully!")


def create_super_admin():
    """Create a super admin user if not exists."""
    from app.core.database import SessionLocal
    
    db = SessionLocal()
    try:
        # Check if super admin already exists
        existing_admin = db.query(Admin).filter(
            Admin.email == settings.SUPER_ADMIN_EMAIL
        ).first()
        
        if existing_admin:
            print(f"⚠️  Super admin already exists: {settings.SUPER_ADMIN_EMAIL}")
            return
        
        # Create super admin
        super_admin = Admin(
            email=settings.SUPER_ADMIN_EMAIL,
            username="superadmin",
            hashed_password=get_password_hash(settings.SUPER_ADMIN_PASSWORD),
            full_name="Super Administrator",
            is_super_admin=True,
            is_active=True
        )
        
        db.add(super_admin)
        db.commit()
        db.refresh(super_admin)
        
        print(f"✅ Super admin created successfully!")
        print(f"   Email: {settings.SUPER_ADMIN_EMAIL}")
        print(f"   Password: {settings.SUPER_ADMIN_PASSWORD}")
        print(f"   ⚠️  Please change the password after first login!")
        
    except Exception as e:
        print(f"❌ Error creating super admin: {str(e)}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    print("🗄️  Initializing Zyntra Database...")
    print()
    
    init_db()
    print()
    
    print("👤 Creating super admin...")
    create_super_admin()
    print()
    
    print("✅ Database initialization complete!")
