from .auth import router as auth_router
from .admins import router as admins_router
from .users import router as users_router
from .gait import user_router as gait_user_router, admin_router as gait_admin_router

__all__ = ["auth_router", "admins_router", "users_router", "gait_user_router", "gait_admin_router"]
