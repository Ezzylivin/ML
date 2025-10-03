# File: auth.py
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import jwt
import os

# This tells FastAPI to look for a "Bearer <token>" in the Authorization header
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# This is our dependency function
def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Decodes the JWT token and returns the user payload.
    This function will be used as a dependency in protected routes.
    """
    JWT_SECRET = os.environ.get("JWT_SECRET")
    if not JWT_SECRET:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="JWT_SECRET is not configured on the server."
        )
        
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # Decode the token using your secret
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        
        # The 'id' in our token corresponds to the user's ID
        user_id: str = payload.get("id")
        if user_id is None:
            raise credentials_exception
            
        # For simplicity, we'll just return the payload.
        # A more complex app might fetch the full user from a database here.
        return payload

    except jwt.PyJWTError:
        raise credentials_exception
