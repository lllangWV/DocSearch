class Document:

    def __init__(self, pages: List[Page]):
        self.pages = pages

    def __len__(self):
        return len(self.pages)
