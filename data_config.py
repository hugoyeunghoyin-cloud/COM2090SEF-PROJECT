# data_config.py
# HKMU Campus Traffic Datasets (Time steps: -30 to 30 mins)
TIME_STEPS = [-30, -20, -10, 0, 10, 20, 30]

CAMPUS_DATA = {
    '1': {
        'name': "IOH Main Campus",
        'traffic': [10, 15, 25, 30, 23, 15, 10],
        'peak': 30
    },
    '2': {
        'name': "MC Mong Kok",
        'traffic': [5, 8, 15, 25, 20, 5, 3],
        'peak': 25
    },
    '3': {
        'name': "JCC Jockey Club",
        'traffic': [6, 9, 18, 28, 25, 10, 5],
        'peak': 28
    }
}
