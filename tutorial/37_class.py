

class CustomL2UString:

    def __init__(self, string):
        self.original = string

    def __len__(self):
        return len(self.original)
    
    def __getitem__(self, idx):
        if idx >= len(self) or idx < -len(self):
            raise IndexError("CustomL2UString index out of range")
        return self.original[idx]
    
    def __repr__(self):
        return f"CustomL2UString, original: {self.original}, length: {len(self)}"
    
    def __call__(self):
        return self.original.upper()
    


if __name__ == "__main__" :

    string = "Hello World"

    custom_string = CustomL2UString(string) ## __init__ 호출
    print(custom_string) ## __repr__ 호출
    print(len(custom_string)) ## __len__ 호출
    print(custom_string[0]) ## __getitem__ 호출
    print(custom_string()) ## __call__ 호출