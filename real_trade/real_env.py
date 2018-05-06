from PyQt5.QAxContainer import *
from PyQt5.QtCore import *
from real_trade.stock import *


class RealEnv(QAxWidget):

    def __init__(self):
        super().__init__()
        self.setControl("KHOPENAPI.KHOpenAPICtrl.1")
        self._set_signal()
        self._log_in()
        self._set_stock()
        self._set_real()
        self._get_account()

        self.recent_TR = time.time()
        self.TR_interval = 0.22

        self.recent_RQ = time.time()
        self.RQ_interval = 4.

    def _log_in(self):
        self.dynamicCall("CommConnect()")
        self.login_event_loop = QEventLoop()
        self.login_event_loop.exec_()

    def _event_connect(self, err_code):
        if err_code == 0:
            print('Successfully Connected')
        else:
            print('Connect Error : ' + str(err_code))
        self.login_event_loop.exit()

    def _set_signal(self):
        self.OnEventConnect.connect(self._event_connect)
        self.OnReceiveRealData.connect(self._receive_real_data)
        self.OnReceiveChejanData.connect(self._receive_chejan_data)

    def _get_code_list(self):
        kospi_list = self.dynamicCall("GetCodeListByMarket(QString)", '0')
        kospi_list = kospi_list.split(';')

        kosdaq_list = self.dynamicCall("GetCodeListByMarket(QString)", '10')
        kosdaq_list = kosdaq_list.split(';')

        return kosdaq_list + kospi_list

    def _set_stock(self):
        self.container = Container()
        code_list = self._get_code_list()
        for code in code_list:
            stock = Stock(code)
            self.container.append_stock(stock)

    def _set_real(self):
        screen_no = 1000
        reg_cnt = 0
        code_list = self._get_code_list()

        fid_list = ''
        for i in range(41, 81):
            if i != 41:
                fid_list += ';'
            fid_list += str(i)

        for code in code_list:
            reg_cnt += 1
            if reg_cnt == 100:
                reg_cnt = 0
                screen_no += 1
            self.dynamicCall("SetRealReg(QString, QString, QString, int)", screen_no, code, fid_list, 1)

    def _get_real_data(self, code, fid):
        return self.dynamicCall("GetCommRealData(QString, QString)", code, fid)

    def _get_account(self):
        acc = self.dynamicCall("GetLoginInfo(QString)", ["ACCNO"])
        self.account = acc.split(';')[0]

    def _receive_real_data(self, code, real_type, real_data):
        if real_type == '주식호가잔량':
            data = dict()
            buy_hoga = list()
            buy_quantity = list()
            sell_hoga = list()
            sell_quantity = list()

            for i in range(10):
                buy_hoga.append(str(abs(int(self._get_real_data(code, 51+i)))))
                buy_quantity.append(str(abs(int(self._get_real_data(code, 71+i)))))
                sell_hoga.append(str(abs(int(self._get_real_data(code, 41+i)))))
                sell_quantity.append(str(abs(int(self._get_real_data(code, 61+i)))))

            data['buy_hoga'] = buy_hoga
            data['buy_quantity'] = buy_quantity
            data['sell_hoga'] = sell_hoga
            data['sell_quantity'] = sell_quantity

            self.container.new_data(code, data, real_type)

        pass

    def get_observation(self):
        pass

    def _set_balance(self):
        self.start_money = 0

    def get_balance(self):
        pass

    def send_order(self, rqname, screen_no, acc_no, order_type, code, quantity, price, hoga, order_no):
        now = time.time()
        if now - self.recent_TR < self.TR_interval:
            time.sleep(self.TR_interval - (now - self.recent_TR))
        self.recent_TR = time.time()

        self.dynamicCall("SendOrder(QString, QString, QString, int, QString, int, int, QString, QString)",
                         [rqname, screen_no, acc_no, order_type, code, quantity, price, hoga, order_no])

    def _receive_chejan_data(self, gubun, item_cnt, fid_list):
        pass

    # environment 의 형태가 잡혀야 구현 가능한 부분은 추후 업데이트