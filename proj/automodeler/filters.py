import logging

class UriFilter(logging.Filter):
    def filter(self, record):
        if hasattr(record, 'request_path') and record.request_path.startswith('/logviewer/'):
            return False