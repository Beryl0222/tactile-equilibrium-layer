"""合成资料与引用的读取检查。"""
import unittest
from inspect_materials import validate

class MaterialTests(unittest.TestCase):
    def test_documents_and_references(self):
        materials,events=validate()
        self.assertEqual(materials['metadata']['source'],'本地合成数据')
        self.assertTrue(events)

if __name__=='__main__':unittest.main()
