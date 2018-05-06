import time


class Stock:

    def __init__(self, code):
        self.code = str(code)
        self.recent_time = time.time()
        self.no_trade = True

        self.buy_hoga = list()
        self.buy_quantity = list()
        self.sell_hoga = list()
        self.sell_quantity = list()

    def new_data(self, data, data_type):
        self.recent_time = time.time()
        self.no_trade = False

        if data_type == '주식호가잔량':
            self.buy_hoga = data['buy_hoga']
            self.buy_quantity = data['buy_quantity']
            self.sell_hoga = data['sell_hoga']
            self.sell_quantity = data['sell_quantity']


class Container:

    def __init__(self):
        self.container = list()

    def append_stock(self, stock):
        self.container.append(stock)

    def new_data(self, code, data, data_type):
        for stock in self.container:
            if stock.code == str(code):
                stock.new_data(data, data_type)
                break
