#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS Programs ChromaDB Query Tool - 使用默认嵌入模型
不需要API密钥，使用ChromaDB自带的默认嵌入函数

作者: Assistant
日期: 2025-09-16
"""

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import json
import os
from pprint import pprint
from typing import List, Dict, Any

class CSProgramsDB:
    """CS项目数据库查询类"""
    
    def __init__(self, db_path: str = "vectors", collection_name: str = "cs_programs_default"):
        """
        初始化数据库连接
        
        Args:
            db_path: 数据库路径
            collection_name: 集合名称
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.embedding_function = None
        
        self._initialize_db()
    
    def _initialize_db(self):
        """初始化数据库和嵌入函数"""
        try:
            # 创建持久化客户端
            self.client = chromadb.PersistentClient(path=self.db_path)
            
            # 使用ChromaDB默认嵌入函数（不需要API）
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
            
            print(f"✅ 成功初始化ChromaDB客户端，数据库路径: {self.db_path}")
            print(f"✅ 使用默认嵌入函数: {type(self.embedding_function).__name__}")
            
        except Exception as e:
            print(f"❌ 数据库初始化失败: {e}")
            raise
    
    def process_admission_data(self, admission_data_str: str) -> str:
        """处理录取数据字符串"""
        if not admission_data_str or admission_data_str == 'null':
            return "暂无录取案例数据"
        
        try:
            # 处理字符串格式的列表转换
            data_str = admission_data_str.replace("'", '"')
            data_list = json.loads(data_str)
            
            cases = []
            for case in data_list:
                case_text = f"录取案例({case.get('录取时间', 'N/A')}): {case.get('录取结果', 'N/A')}。"
                case_text += f"申请者背景: {case.get('学校（档次）', 'N/A')} {case.get('本科专业', 'N/A')}专业，"
                case_text += f"GPA/Rank {case.get('GPA/Rank', 'N/A')}，"
                case_text += f"科研经历 {case.get('科研经历', 'N/A')}，"
                case_text += f"实习经历 {case.get('实习经历', 'N/A')}。"
                if case.get('其他（语言/推荐信）'):
                    case_text += f"其他信息: {case.get('其他（语言/推荐信）', '')}"
                cases.append(case_text)
            
            return " ".join(cases)
        except (json.JSONDecodeError, TypeError, AttributeError):
            return f"录取数据: {admission_data_str}"
    
    def create_document_text(self, row: pd.Series) -> str:
        """创建用于语义搜索的富文本文档"""
        admission_text = self.process_admission_data(row['admission_data'])
        
        document = f"""
项目名称: {row['program_name']}
所属大学: {row['university']}
地区: {row['region']}
项目等级: {row['tier']}
学制: {row['duration']}
授课语言: {row['language']}
学位类型: {row['degree_type']}

项目优点: {row['pros']}

项目缺点: {row['cons']}

招生偏好: {row['admission_preference']}

申请注意事项: {row['application_notes']}

奖学金信息: {row['scholarship']}

过往录取案例: {admission_text}

其他信息: {row['other_info']} {row['other_notes']}
        """
        return document.strip()
    
    def load_and_process_data(self, csv_path: str = "global_cs_programs.csv") -> tuple:
        """加载和处理CSV数据"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
        
        # 读取CSV数据
        df = pd.read_csv(csv_path).fillna('')
        print(f"📊 读取到 {len(df)} 个CS项目")
        
        documents = []
        metadatas = []
        ids = []
        
        print("🔄 处理项目数据...")
        
        for index, row in df.iterrows():
            # 创建文档文本
            doc_text = self.create_document_text(row)
            documents.append(doc_text)
            
            # 创建元数据
            metadata = {
                'program_name': str(row['program_name']),
                'university': str(row['university']),
                'region': str(row['region']),
                'tier': str(row['tier']),
                'duration': str(row['duration']),
                'language': str(row['language']),
                'degree_type': str(row['degree_type']),
                'internship_required': True if str(row['internship_required']).strip() == '是' else False,
                'thesis_required': True if str(row['thesis_required']).strip() == '是' else False,
                'admission_data_count': int(row['admission_data_count']) if str(row['admission_data_count']).isdigit() else 0
            }
            metadatas.append(metadata)
            
            # 创建唯一ID
            ids.append(f"program_{index}")
        
        print(f"✅ 成功处理 {len(documents)} 个项目")
        return documents, metadatas, ids
    
    def create_collection(self, documents: List[str], metadatas: List[Dict], ids: List[str], 
                         recreate: bool = True) -> None:
        """创建或重新创建集合"""
        
        # 删除现有集合（如果存在且需要重新创建）
        if recreate:
            try:
                self.client.delete_collection(name=self.collection_name)
                print(f"🗑️ 删除现有集合: {self.collection_name}")
            except:
                print(f"ℹ️ 集合 {self.collection_name} 不存在，将创建新集合")
        
        # 创建新集合
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function
        )
        print(f"✅ 创建集合: {self.collection_name}")
        
        # 批量添加数据
        batch_size = 10
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        print(f"📦 分 {total_batches} 批次添加数据，每批 {batch_size} 个项目")
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            batch_metadata = metadatas[i:i+batch_size]
            
            batch_num = (i // batch_size) + 1
            
            try:
                self.collection.add(
                    documents=batch_docs,
                    metadatas=batch_metadata,
                    ids=batch_ids
                )
                print(f"✅ 批次 {batch_num}/{total_batches} 添加成功")
                
            except Exception as e:
                print(f"❌ 批次 {batch_num} 添加失败: {str(e)}")
                break
        
        print(f"🎉 集合创建完成，包含 {self.collection.count()} 个项目")
    
    def get_existing_collection(self) -> bool:
        """获取现有集合"""
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            print(f"✅ 连接到现有集合 '{self.collection_name}'，包含 {self.collection.count()} 个项目")
            return True
        except ValueError:
            print(f"⚠️ 集合 '{self.collection_name}' 不存在")
            available_collections = self.client.list_collections()
            if available_collections:
                print("📋 可用集合:", [col.name for col in available_collections])
            return False
    
    def query_programs(self, query_text: str, n_results: int = 5, 
                      where_filter: Dict = None) -> Dict[str, Any]:
        """查询项目"""
        if not self.collection:
            raise ValueError("集合未初始化，请先创建或连接到集合")
        
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where_filter
            )
            return results
        except Exception as e:
            print(f"❌ 查询失败: {e}")
            return {}
    
    def print_query_results(self, results: Dict[str, Any], query_description: str = ""):
        """打印查询结果"""
        if not results or not results.get('documents') or not results['documents'][0]:
            print(f"\n🤷 {query_description} - 未找到符合条件的结果")
            return
        
        print(f"\n🔍 {query_description}")
        print("=" * 50)
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            print(f"\n📋 结果 {i+1} (相似度: {1-distance:.3f}):")
            print(f"  项目: {metadata.get('program_name', 'N/A')}")
            print(f"  大学: {metadata.get('university', 'N/A')}")
            print(f"  地区: {metadata.get('region', 'N/A')}")
            print(f"  等级: {metadata.get('tier', 'N/A')}")
            print(f"  学制: {metadata.get('duration', 'N/A')}")
            print(f"  语言: {metadata.get('language', 'N/A')}")
            if metadata.get('admission_data_count', 0) > 0:
                print(f"  录取案例数: {metadata['admission_data_count']}")
    
    def get_collection_stats(self):
        """获取集合统计信息"""
        if not self.collection:
            print("❌ 集合未初始化")
            return
        
        all_data = self.collection.get()
        total_count = len(all_data['metadatas'])
        
        print(f"\n📊 集合统计信息")
        print("=" * 30)
        print(f"总项目数: {total_count}")
        
        # 按地区统计
        regions = [meta['region'] for meta in all_data['metadatas']]
        region_counts = {}
        for region in regions:
            region_counts[region] = region_counts.get(region, 0) + 1
        
        print(f"\n📍 按地区分布:")
        for region, count in sorted(region_counts.items()):
            print(f"  {region}: {count} 个项目")
        
        # 按等级统计
        tiers = [meta['tier'] for meta in all_data['metadatas']]
        tier_counts = {}
        for tier in tiers:
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        print(f"\n🏆 按等级分布:")
        for tier, count in sorted(tier_counts.items()):
            print(f"  {tier}: {count} 个项目")
        
        # 论文要求统计
        thesis_required = sum(1 for meta in all_data['metadatas'] if meta['thesis_required'])
        print(f"\n📝 需要论文的项目: {thesis_required}/{total_count}")


def main():
    """主函数 - 演示用法"""
    print("🚀 CS Programs ChromaDB Query Tool (默认嵌入模型)")
    print("=" * 60)
    
    # 初始化数据库
    db = CSProgramsDB()
    
    # 尝试连接现有集合
    if not db.get_existing_collection():
        print("\n🔄 未找到现有集合，开始创建新集合...")
        
        # 加载和处理数据
        try:
            documents, metadatas, ids = db.load_and_process_data()
            
            # 创建集合
            db.create_collection(documents, metadatas, ids)
            
        except FileNotFoundError as e:
            print(f"❌ {e}")
            print("请确保 'global_cs_programs.csv' 文件在当前目录下")
            return
        except Exception as e:
            print(f"❌ 数据处理失败: {e}")
            return
    
    # 显示统计信息
    db.get_collection_stats()
    
    # 执行示例查询
    print(f"\n🔍 执行示例查询")
    print("=" * 40)
    
    # 查询1: 短学制项目
    results1 = db.query_programs(
        query_text="学制短 快速毕业 时间紧凑",
        n_results=5,
        where_filter={"duration": {"$in": ["9个月", "10个月", "11个月", "12个月", "1年", "一年"]}}
    )
    db.print_query_results(results1, "短学制项目 (1年以内)")
    
    # 查询2: 顶级项目
    results2 = db.query_programs(
        query_text="顶级大学 计算机科学 人工智能",
        n_results=5,
        where_filter={"tier": {"$in": ["T0", "T1"]}}
    )
    db.print_query_results(results2, "顶级项目 (T0/T1)")
    
    # 查询3: 英国项目
    results3 = db.query_programs(
        query_text="英国 录取友好 申请建议",
        n_results=3,
        where_filter={"region": "英国"}
    )
    db.print_query_results(results3, "英国CS项目")
    
    # 查询4: 不需要论文的项目
    results4 = db.query_programs(
        query_text="不需要论文 coursework 授课型",
        n_results=3,
        where_filter={"thesis_required": False}
    )
    db.print_query_results(results4, "不需要论文的项目")
    
    print(f"\n✅ 查询演示完成!")
    print(f"\n💡 使用提示:")
    print("- 可以修改 main() 函数中的查询来测试不同搜索")
    print("- 支持中文和英文查询")
    print("- 可以使用 where_filter 进行精确筛选")
    print("- 默认嵌入模型无需API密钥，但效果可能不如专业模型")


if __name__ == "__main__":
    main()
