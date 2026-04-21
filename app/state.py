
# Shared in-memory store
# Holds the uploaded dataframe and the trained model artifact between requests.
# In production you'd replace this with a proper session/database layer.

app_state: dict = {
    "dataframe":        None,   # pd.DataFrame uploaded by the user
    "trained_artifact": None,   # dict with model, preprocessor, metadata
}