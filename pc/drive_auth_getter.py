from pydrive2.auth import GoogleAuth

gauth = GoogleAuth()
gauth.OAuthSettingsFile = "client_secrets.json"
gauth.LocalWebserverAuth()  # This will open a web browser for authentication
gauth.SaveCredentialsFile("credentials.json")
