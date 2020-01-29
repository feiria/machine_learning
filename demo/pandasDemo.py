import pandas
from IPython.core.display import display

data = {
    "name": ["小于", "小寒", "小鱼", "小涵"],
    "city": ["北京", "上海", "广州", "深圳"],
    "age": ["18", "20", "22", "34"],
    "height": ["162", "161", "165", "166"]
}
data_frame = pandas.DataFrame(data)
display(data_frame[data_frame.city != "北京"])
