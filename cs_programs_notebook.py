# Cell 1: Import libraries and setup
import pandas as pd
import polars as pl
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from pprint import pprint
import json
import re

print("Libraries imported successfully!")

# Cell 2: Read CS programs CSV data
# Using pandas for better JSON handling, then convert to polars if needed
df = pd.read_csv("global_cs_programs.csv").fillna('')

print(f"DataFrame shape: {df.shape}")
print("\nColumn names:")
pprint(list(df.columns))
print("\nFirst 2 rows preview:")
print(df[['program_name', 'university', 'region', 'tier']].head(2))

# Cell 3: Initialize ChromaDB with persistent storage
chroma_client = chromadb.PersistentClient(path="vectors")

# Initialize Google Gemini embedding function
google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key="API_KEY",
    model_name="gemini-embedding-001"
)

print("ChromaDB and embedding function initialized!")

# Cell 4: Data preprocessing and document creation
def process_admission_data(admission_data_str):
    """Convert admission_data string to readable text"""
    if not admission_data_str or admission_data_str == 'null':
        return "暂无录取案例数据"
    
    try:
        # Handle string format list conversion
        data_str = admission_data_str.replace("'", "\"")
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

def create_document_text(row):
    """Create rich document text for semantic search"""
    admission_text = process_admission_data(row['admission_data'])
    
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

print("Data processing functions defined!")

# Cell 5: Process all data for ChromaDB
documents = []
metadatas = []
ids = []

print(f"Processing {len(df)} CS programs...")

for index, row in df.iterrows():
    # Create document text
    doc_text = create_document_text(row)
    documents.append(doc_text)
    
    # Create metadata for filtering
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
    
    # Create unique ID
    ids.append(f"program_{index}")

print(f"Processed {len(documents)} programs successfully!")

# Cell 6: Create ChromaDB collection
collection_name = "cs_programs_collection"

# Delete existing collection if it exists
try:
    chroma_client.delete_collection(name=collection_name)
    print(f"Deleted existing collection: {collection_name}")
except:
    print("No existing collection to delete")

# Create new collection with embedding function
collection = chroma_client.create_collection(
    name=collection_name,
    embedding_function=google_ef
)

print(f"Created collection: {collection_name}")

# Cell 7: Add data to collection in batches
batch_size = 10
total_batches = (len(documents) + batch_size - 1) // batch_size

print(f"Adding {len(documents)} programs in {total_batches} batches of {batch_size}")

for i in range(0, len(documents), batch_size):
    batch_docs = documents[i:i+batch_size]
    batch_ids = ids[i:i+batch_size]
    batch_metadata = metadatas[i:i+batch_size]
    
    batch_num = (i // batch_size) + 1
    
    print(f"Processing batch {batch_num}/{total_batches}")
    
    try:
        collection.add(
            documents=batch_docs,
            metadatas=batch_metadata,
            ids=batch_ids
        )
        print(f"✓ Added batch {batch_num}")
        
    except Exception as e:
        print(f"✗ Error in batch {batch_num}: {str(e)}")
        break

print(f"\nCollection created with {collection.count()} programs")

# Cell 8: Test semantic search queries with filters
print("\n=== Testing Semantic Search with Filters ===")

# Query 1: Find TOP-TIER programs suitable for mainland Chinese students
results1 = collection.query(
    query_texts=["陆本背景申请 录取案例 友好"],
    n_results=5,
    where={"$and": [{"tier": {"$in": ["T0", "T1"]}}, {"admission_data_count": {"$gt": 0}}]}
)

print("\n🔍 Query: 'T0/T1级别且有陆本录取案例的项目'")
for i, (doc, metadata, distance) in enumerate(zip(
    results1['documents'][0], 
    results1['metadatas'][0], 
    results1['distances'][0]
)):
    print(f"\n📋 Result {i+1} (相似度: {1-distance:.3f}):")
    print(f"项目: {metadata['program_name']}")
    print(f"大学: {metadata['university']}")
    print(f"地区: {metadata['region']}")
    print(f"等级: {metadata['tier']}")
    print(f"学制: {metadata['duration']}")
    print(f"录取案例数: {metadata['admission_data_count']}")

# Cell 9: More targeted queries with duration filter
# Query 2: Short duration programs (filter for programs ≤ 12 months)
results2 = collection.query(
    query_texts=["学制短 快速毕业 时间紧凑"],
    n_results=5,
    where={"duration": {"$in": ["9个月", "10个月", "11个月", "12个月", "1年", "一年"]}}
)

print("\n🔍 Query: '1年以内的短学制优质项目'")
for i, (doc, metadata, distance) in enumerate(zip(
    results2['documents'][0], 
    results2['metadatas'][0], 
    results2['distances'][0]
)):
    print(f"\n📋 Result {i+1}:")
    print(f"项目: {metadata['program_name']}")
    print(f"学制: {metadata['duration']}")
    print(f"大学: {metadata['university']}")
    print(f"等级: {metadata['tier']}")
    print(f"地区: {metadata['region']}")

# Query 3: Scholarship opportunities (exclude programs with "较难获得" scholarships)
results3 = collection.query(
    query_texts=["奖学金 资助 经济支持 容易申请"],
    n_results=5
)

print("\n🔍 Query: '有奖学金机会的优质项目'")
for i, (doc, metadata, distance) in enumerate(zip(
    results3['documents'][0], 
    results3['metadatas'][0], 
    results3['distances'][0]
)):
    print(f"\n📋 Result {i+1}:")
    print(f"项目: {metadata['program_name']}")
    print(f"大学: {metadata['university']}")
    print(f"地区: {metadata['region']}")
    print(f"等级: {metadata['tier']}")

# Cell 10: Advanced filtering with metadata
print("\n=== Advanced Filtering ===")

# Filter by region and tier with better query
uk_t0_results = collection.query(
    query_texts=["计算机科学 人工智能 机器学习 顶级项目"],
    n_results=5,
    where={"$and": [{"region": "英国"}, {"tier": "T0"}]}
)

# Additional query: US programs with good admission data
us_programs = collection.query(
    query_texts=["英国 录取友好 申请建议"],
    n_results=3,
    where={"$and": [{"region": "英国"}, {"admission_data_count": {"$gt": 0}}]}
)

print("\n🎯 Filter: 英国T0级别项目")
for i, (doc, metadata, distance) in enumerate(zip(
    uk_t0_results['documents'][0], 
    uk_t0_results['metadatas'][0], 
    uk_t0_results['distances'][0]
)):
    print(f"\n📋 Result {i+1}:")
    print(f"项目: {metadata['program_name']}")
    print(f"大学: {metadata['university']}")
    print(f"学位类型: {metadata['degree_type']}")
    print(f"是否需要论文: {'是' if metadata['thesis_required'] else '否'}")

print("\n🎯 Filter: 英国有录取数据的项目")
for i, (doc, metadata, distance) in enumerate(zip(
    us_programs['documents'][0], 
    us_programs['metadatas'][0], 
    us_programs['distances'][0]
)):
    print(f"\n📋 Result {i+1}:")
    print(f"项目: {metadata['program_name']}")
    print(f"大学: {metadata['university']}")
    print(f"等级: {metadata['tier']}")
    print(f"录取案例数: {metadata['admission_data_count']}")

# Cell 11: Specialized queries for different needs
print("\n=== Specialized Queries ===")

# Query for programs without thesis requirement
no_thesis_programs = collection.query(
    query_texts=["不需要论文 coursework 授课型"],
    n_results=3,
    where={"$and": [{"thesis_required": False}, {"tier": {"$in": ["T0", "T1"]}}]}
)

print("\n🎯 不需要论文的顶级项目:")
for i, (doc, metadata, distance) in enumerate(zip(
    no_thesis_programs['documents'][0], 
    no_thesis_programs['metadatas'][0], 
    no_thesis_programs['distances'][0]
)):
    print(f"\n📋 Result {i+1}:")
    print(f"项目: {metadata['program_name']}")
    print(f"大学: {metadata['university']}")
    print(f"学制: {metadata['duration']}")
    print(f"等级: {metadata['tier']}")

# Query for English-taught programs in non-English countries
non_english_countries = collection.query(
    query_texts=["英语授课 国际项目"],
    n_results=3,
    where={"$and": [
        {"language": "EN"}, 
        {"region": {"$nin": ["英国", "澳洲", "加拿大"]}}
    ]}
)

print("\n🎯 非英语国家的英语授课项目:")
for i, (doc, metadata, distance) in enumerate(zip(
    non_english_countries['documents'][0], 
    non_english_countries['metadatas'][0], 
    non_english_countries['distances'][0]
)):
    print(f"\n📋 Result {i+1}:")
    print(f"项目: {metadata['program_name']}")
    print(f"大学: {metadata['university']}")
    print(f"地区: {metadata['region']}")
    print(f"语言: {metadata['language']}")

# Cell 12: Collection statistics and analysis
print("\n=== Collection Statistics ===")
print(f"总项目数: {collection.count()}")

# Get all data for analysis
all_data = collection.get()

# Analyze by region
regions = [meta['region'] for meta in all_data['metadatas']]
region_counts = {}
for region in regions:
    region_counts[region] = region_counts.get(region, 0) + 1

print("\n📊 按地区分布:")
for region, count in sorted(region_counts.items()):
    print(f"  {region}: {count} 个项目")

# Analyze by tier
tiers = [meta['tier'] for meta in all_data['metadatas']]
tier_counts = {}
for tier in tiers:
    tier_counts[tier] = tier_counts.get(tier, 0) + 1

print("\n📊 按等级分布:")
for tier, count in sorted(tier_counts.items()):
    print(f"  {tier}: {count} 个项目")

# Analyze thesis requirements
thesis_required = sum(1 for meta in all_data['metadatas'] if meta['thesis_required'])
print(f"\n📊 需要论文的项目: {thesis_required}/{len(all_data['metadatas'])}")

print("\n✅ CS Programs ChromaDB setup complete!")
