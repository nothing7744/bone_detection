target=[0,1,1,0,1,1,1,0,0]
classes = [1 if obj==0 else 2 for obj in target]
print(classes)