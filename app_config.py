class Config:
    DEBUG = True
    DEVELOPMENT = True
    MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB
    SECRET_KEY = 'SUPER SECRET KEY WOW'
    FLASK_SECRET = SECRET_KEY


class ProductionConfig(Config):
    DEBUG = False
    DEVELOPMENT = False
