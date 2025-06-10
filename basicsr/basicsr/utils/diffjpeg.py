# Dummy patch to disable DiffJPEG import error on Streamlit Cloud

class DiffJPEG:
    def __init__(self, *args, **kwargs):
        pass

    def forward(self, x):
        return x
