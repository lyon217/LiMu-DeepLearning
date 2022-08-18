class Student:
    type = "person"
    class_num = 34

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def get_age(self):
        print(self.__class__.__name__)  # 注意这里
        return self.age

    @classmethod
    def get_teacher(self):
        return "Brain"


print(Student.__class__.__name__)  # 注意这里
print("*" * 30)
stu = Student("weihua", 18)
print(stu.__class__.__name__)
