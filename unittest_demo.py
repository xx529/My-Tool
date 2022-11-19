import unittest
from unittestreport import ddt, list_data, json_data, yaml_data


class TestDemo(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_success(self):
        """
        这个是成功用例
        """

        self.assertEqual(2, 2)

    def test_fail(self):
        """
        这个是失败用例
        """
        self.assertTrue(0)

    def test_error(self):
        """
        这个是错误用例
        """

        raise Exception('error')


@ddt
class TestSomething(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @list_data(your_list_obj)
    def test_1(self, data):
        """
        测试列表形式用例
        """

        num = data['num']

        self.assertEqual(num, 0)

    @json_data('./data/test.json')
    def test_2(self, data):
        """
        测试json文件用例
        """

        num = data['num']

        self.assertEqual(num, 0)

    @yaml_data('./data/test.yaml')
    def test_3(self, data):
        """
        测试yaml文件用例
        """

        num1 = data['num1']
        num2 = data['num2']

        self.assertGreater(num1, num2)
