# Copyright (c) 2014-2016 CNLAB of KMUST
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import logging
try:
    import queue as Queue # module re-named in Python 3
except ImportError:
    import Queue
import sqlite3
import os

logger = logging.getLogger('muser')


#global var
class MuserDB(object):
    __instance = None
    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super(MuserDB,cls).__new__(cls)
            cls.__instance.__initialized = False
        return cls.__instance

    def __init__(self):
        if(self.__initialized): return
        self.__initialized = True

    def setdb(self, file= None, table=None, debug = False):
        if os.path.exists(file) and os.path.isfile(file):
            self.DB_FILE_PATH = file
            self.TABLE_NAME = table
            self.debug = debug
        self.conn = None
        self.cursor = None

    def connect(self):
        try:
            self.conn = sqlite3.connect(self.DB_FILE_PATH)
        except:
            self.conn = None

    def get_cursor(self):
        if self.conn is not None:
            return self.conn.cursor()
        else:
            return None

    def drop_table(self, conn, table):
        if table is not None and table != '':
            sql = 'DROP TABLE IF EXISTS ' + table
            logger.debug('Execute sql:[{}]'.format(sql))
            cu = self.conn.cursor()
            cu.execute(sql)
            conn.commit()
            logger.debug('Drop table [{}] successfully.'.format(table))
            self.close_all(conn, cu)
        else:
            print('the [{}] is empty or equal None!'.format(sql))

    def create_table(self, conn, sql):
        if sql is not None and sql != '':
            cu = self.conn.cursor()
            logger.debug('Execute sql:[{}]'.format(sql))
            cu.execute(sql)
            conn.commit()
            #logger.debug('Create table [{}] successfully.'.format(table))
            self.close_all(conn, cu)
        else:
            print('the [{}] is empty or equal None!'.format(sql))

    def close(self):
        try:
            if self.conn is not None:
                self.conn.close()
        finally:
            if self.conn is not None:
                self.conn.close()

    def save(self, sql, data):
        if sql is not None and sql != '':
            if data is not None:
                for d in data:
                    logger.debug('Execute sql:[{}],Arguments:[{}]'.format(sql, d))
                    self.cursor.execute(sql, d)
                    self.conn.commit()
                self.close_all(self.conn, self.cursor)
        else:
            print('the [{}] is empty or equal None!'.format(sql))

    # def fetchall(self, conn, sql):
    #     if sql is not None and sql != '':
    #         cu = self.get_cursor(conn)
    #         logger.debug('Execute sql:[{}]'.format(sql))
    #         cu.execute(sql)
    #         r = cu.fetchall()
    #         if len(r) > 0:
    #             for e in range(len(r)):
    #                 print(r[e])
    #     else:
    #         print('the [{}] is empty or equal None!'.format(sql))

    def query(self, sql, data=None):
        logger.debug('Execute sql:[{}], Arguments:[{}]'.format(sql, data))
        if sql is not None and sql != '':
            #Do this instead
            cu = self.conn.cursor()
            logger.debug('Execute sql:[{}], Arguments:[{}]'.format(sql, data))
            cu.execute(sql, data)
            r = cu.fetchall()
            return r
        else:
            logger.debug('the [{}] is empty or equal None!'.format(sql))

    def update(self, sql, data):
        if sql is not None and sql != '':
            if data is not None:
                cu = set.conn.cursor()
                for d in data:
                    logger.debug('Execute sql:[{}], Arguments:[{}]'.format(sql, d))
                    cu.execute(sql, d)
                    self.conn.commit()
                self.close_all(self.conn, cu)
        else:
            logger.debug('the [{}] is empty or equal None!'.format(sql))

    def delete(self, conn, sql, data):
        if sql is not None and sql != '':
            if data is not None:
                cu = self.get_cursor(conn)
                for d in data:
                    logger.debug('Execute sql:[{}],Arguments:[{}]'.format(sql, d))
                    cu.execute(sql, d)
                    conn.commit()
                self.close_all(conn, cu)
        else:
            logger.debug('the [{}] is empty or equal None!'.format(sql))

# def drop_table_test():
#     conn = get_conn(DB_FILE_PATH)
#     drop_table(conn, TABLE_NAME)
#
# def create_table_test():
#     create_table_sql = '''CREATE TABLE `student` (
#                           `id` int(11) NOT NULL,
#                           `name` varchar(20) NOT NULL,
#                           `gender` varchar(4) DEFAULT NULL,
#                           `age` int(11) DEFAULT NULL,
#                           `address` varchar(200) DEFAULT NULL,
#                           `phone` varchar(20) DEFAULT NULL,
#                            PRIMARY KEY (`id`)
#                         )'''
#     conn = get_conn(DB_FILE_PATH)
#     create_table(conn, create_table_sql)
##
# def fetchall_test():
#     fetchall_sql = '''SELECT * FROM student'''
#     conn = get_conn(DB_FILE_PATH)
#     fetchall(conn, fetchall_sql)
#
# def fetchone_test():
#     fetchone_sql = 'SELECT * FROM student WHERE ID = ? '
#     data = 1
#     conn = get_conn(DB_FILE_PATH)
#     fetchone(conn, fetchone_sql, data)
#
# def update_test():
#     update_sql = 'UPDATE student SET name = ? WHERE ID = ? '
#     data = [('HongtenAA', 1),
#             ('HongtenBB', 2),
#             ('HongtenCC', 3),
#             ('HongtenDD', 4)]
#     conn = get_conn(DB_FILE_PATH)
#     update(conn, update_sql, data)
#
# def delete_test():
#     delete_sql = 'DELETE FROM student WHERE NAME = ? AND ID = ? '
#     data = [('HongtenAA', 1),
#             ('HongtenCC', 3)]
#     conn = get_conn(DB_FILE_PATH)
#     delete(conn, delete_sql, data)
#
#
# def init():
#     global DB_FILE_PATH
#     DB_FILE_PATH = 'c:\\test\\hongten.db'
#     global TABLE_NAME
#     TABLE_NAME = 'student'
#     global debug
#     debug = True
#     print('debug : {}'.format(debug))
#     drop_table_test()
#     create_table_test()
#     save_test()
#
#
# def main():
#     init()
#     fetchall_test()
#     print('#' * 50)
#     fetchone_test()
#     print('#' * 50)
#     update_test()
#     fetchall_test()
#     print('#' * 50)
#     delete_test()
#     fetchall_test()
#
# if __name__ == '__main__':
#     main()

dbfile = None
if os.environ.has_key('MUSERHOME'):
    homedir = os.getenv('MUSERHOME')
    dbfile = homedir+os.path.sep+'db'+os.path.sep+'muser.db'

if dbfile is not None and os.path.exists(dbfile):
    musersqlite = MuserDB()
    musersqlite.setdb(file = dbfile)
    # muserdb.connect()
    # data = ['observatory']
    # sql = '''select * from t_global where keyName=?''' #self, sql, data
    # print muserdb.query(sql,data)
    # muserdb.close()
else:
    musersqlite = None

