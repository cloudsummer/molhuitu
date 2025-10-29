"""
增量超图加载器 - 基于bonds-only数据构建完整超图
遵循KISS原则，直接复用现有的超边构建逻辑
"""

import torch
import pickle
import hashlib
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import rdBase
from torch_geometric.data import Data

# 导入现有的超边构建函数
from .hypergraph_construction import (
    get_ring_features, 
    identify_all_functional_groups,
    compute_unified_hyperedge_dim,
    ensure_canonical_feature_dims,
    pad_to_unified_dim
)
from .hydrogen_bonds import HydrogenBondIdentifier

logger = logging.getLogger(__name__)


class SimpleGraphDataset:
    """简单的图数据集包装类，兼容现有的训练流程"""
    
    def __init__(self, graphs: List[Data]):
        self.graphs = graphs
    
    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx: int) -> Data:
        return self.graphs[idx]
    
    def get_feature_dim(self) -> int:
        """获取节点特征维度"""
        if len(self.graphs) > 0:
            return self.graphs[0].x.size(1)
        return 0
    
    def get_stats(self) -> Dict:
        """获取数据集统计信息"""
        stats = {
            'num_graphs': len(self.graphs),
            'num_nodes': [],
            'num_edges': [],
            'feature_dim': self.get_feature_dim()
        }
        
        for g in self.graphs:
            stats['num_nodes'].append(g.x.size(0))
            if hasattr(g, 'hyperedge_index') and g.hyperedge_index.numel() > 0:
                num_edges = int(g.hyperedge_index[1].max().item() + 1)
                stats['num_edges'].append(num_edges)
            else:
                stats['num_edges'].append(0)
        
        # 计算统计量
        stats['avg_nodes'] = np.mean(stats['num_nodes'])
        stats['avg_edges'] = np.mean(stats['num_edges'])
        stats['std_nodes'] = np.std(stats['num_nodes'])
        stats['std_edges'] = np.std(stats['num_edges'])
        
        return stats


class IncrementalHypergraphLoader:
    """增量超图加载器 - 从bonds-only数据构建完整超图（支持分片并行）"""
    
    def __init__(self, bonds_data_dir: str, config: Dict, 
                 cache_dir: str = "hydra/version2/cache",
                 shard_index: Optional[int] = None, 
                 total_shards: Optional[int] = None):
        self.bonds_data_dir = Path(bonds_data_dir)
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.target_types = config.get('hypergraph_types', ['bond'])
        
        # 分片参数
        self.shard_index = shard_index
        self.total_shards = total_shards
        self.is_sharded = shard_index is not None and total_shards is not None
        
        # 生成缓存键（包含分片信息）
        self.cache_key = self._make_cache_key()
        
        # 确保缓存目录存在
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Incremental loader initialized for types: {self.target_types}")
        if self.is_sharded:
            logger.info(f"Sharded mode: processing shard {shard_index}/{total_shards}")
        logger.info(f"Cache key: {self.cache_key}")
    
    def load_dataset(self) -> SimpleGraphDataset:
        """加载或构建增量超图数据集（支持分片并行）"""
        if self.is_sharded:
            # 分片模式：处理单个分片
            return self._load_shard_dataset()
        else:
            # 非分片模式：尝试合并分片或直接处理
            return self._load_unified_dataset()
    
    def _load_shard_dataset(self) -> SimpleGraphDataset:
        """加载单个分片的数据集"""
        shard_cache_file = self.cache_dir / f"incremental_{self.cache_key}.pt"
        
        if shard_cache_file.exists():
            logger.info(f"Loading cached shard {self.shard_index}: {shard_cache_file}")
            with open(shard_cache_file, 'rb') as f:
                graphs = torch.load(f, weights_only=False)
            return SimpleGraphDataset(graphs)
        
        logger.info(f"Building incremental hypergraph dataset for shard {self.shard_index}...")
        graphs = self._build_incremental()
        
        # 缓存分片结果
        logger.info(f"Caching shard {self.shard_index} to: {shard_cache_file}")
        torch.save(graphs, shard_cache_file)
        
        return SimpleGraphDataset(graphs)
    
    def _load_unified_dataset(self) -> SimpleGraphDataset:
        """加载统一的数据集（尝试从分片合并或直接构建）"""
        base_cache_key = self._make_base_cache_key()
        unified_cache_file = self.cache_dir / f"incremental_{base_cache_key}.pt"
        
        # 检查是否存在统一缓存
        if unified_cache_file.exists():
            logger.info(f"Loading cached unified dataset: {unified_cache_file}")
            with open(unified_cache_file, 'rb') as f:
                graphs = torch.load(f, weights_only=False)
            return SimpleGraphDataset(graphs)
        
        # 尝试从分片合并
        merged_graphs = self._try_merge_shards()
        if merged_graphs is not None:
            logger.info(f"Successfully merged {len(merged_graphs)} graphs from shards")
            # 缓存合并结果
            torch.save(merged_graphs, unified_cache_file)
            return SimpleGraphDataset(merged_graphs)
        
        # 没有分片，直接构建
        logger.info("Building incremental hypergraph dataset...")
        graphs = self._build_incremental()
        
        # 缓存结果
        logger.info(f"Caching unified dataset to: {unified_cache_file}")
        torch.save(graphs, unified_cache_file)
        
        return SimpleGraphDataset(graphs)
    
    def _build_incremental(self) -> List[Data]:
        """增量构建超图 - 在bonds基础上添加其他超边"""
        # 1. 加载bonds-only数据
        bonds_graphs = self._load_bonds_data()
        logger.info(f"Loaded {len(bonds_graphs)} bonds-only graphs")
        
        # 2. 如果只需要bond，直接返回
        if self.target_types == ['bond']:
            logger.info("Only bonds required, returning original data")
            return bonds_graphs
        
        # 3. 加载global_stats
        global_stats = self._load_global_stats()
        
        # 4. 为每个分子增量添加超边
        enhanced_graphs = []
        failed_count = 0
        
        for bond_data in tqdm(bonds_graphs, desc="Adding incremental hyperedges"):
            try:
                enhanced_data = self._add_incremental_hyperedges(bond_data, global_stats)
                if enhanced_data is not None:
                    enhanced_graphs.append(enhanced_data)
                else:
                    failed_count += 1
            except Exception as e:
                logger.warning(f"Failed to process molecule {getattr(bond_data, 'id', 'unknown')}: {e}")
                failed_count += 1
        
        logger.info(f"Successfully enhanced {len(enhanced_graphs)} graphs, failed: {failed_count}")
        return enhanced_graphs
    
    def _add_incremental_hyperedges(self, bond_data: Data, global_stats: Dict) -> Optional[Data]:
        """为单个分子添加增量超边"""
        try:
            # 1. 从bonds数据重建分子对象
            mol = self._reconstruct_molecule(bond_data)
            if mol is None:
                return None
            
            # 2. 提取现有的bonds超边信息
            existing_hyperedge_data = self._extract_bonds_hyperedges(bond_data)
            
            # 3. 计算需要添加的超边类型
            additional_types = [t for t in self.target_types if t != 'bond']
            
            # 4. 增量添加每种超边类型
            for edge_type in additional_types:
                new_edges = self._compute_hyperedges_by_type(mol, edge_type, global_stats)
                existing_hyperedge_data.extend(new_edges)
            
            # 5. 重新组装Data对象
            return self._reassemble_data_object(bond_data, existing_hyperedge_data)
            
        except Exception as e:
            logger.warning(f"Error adding incremental hyperedges for {getattr(bond_data, 'id', 'unknown')}: {e}")
            return None
    
    def _reconstruct_molecule(self, bond_data: Data) -> Optional[Chem.Mol]:
        """从bonds数据重建RDKit分子对象"""
        try:
            # Suppress RDKit warnings
            rdBase.DisableLog('rdApp.warning')
            rdBase.DisableLog('rdApp.error')
            
            mol = Chem.MolFromSmiles(bond_data.smiles)
            if mol is None:
                return None
            
            # 添加氢原子并生成3D结构（简化版）
            mol = Chem.AddHs(mol)
            
            # 尝试生成3D结构，失败则使用2D
            try:
                if AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=50) < 0:
                    AllChem.Compute2DCoords(mol)
            except:
                AllChem.Compute2DCoords(mol)
            
            # 力场优化（可选）
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except:
                pass
            
            return mol
            
        except Exception as e:
            logger.debug(f"Molecule reconstruction failed: {e}")
            return None
        finally:
            # Re-enable RDKit logging
            rdBase.EnableLog('rdApp.warning')
            rdBase.EnableLog('rdApp.error')
    
    def _extract_bonds_hyperedges(self, bond_data: Data) -> List[Dict]:
        """从bonds数据提取现有的bonds超边信息"""
        hyperedge_data = []
        
        # 从hyperedge_index重建超边列表
        if hasattr(bond_data, 'hyperedge_index') and bond_data.hyperedge_index.numel() > 0:
            # 获取每个超边包含的节点
            num_hyperedges = int(bond_data.hyperedge_index[1].max().item() + 1)
            
            for i in range(num_hyperedges):
                # 找到属于这个超边的所有节点
                mask = bond_data.hyperedge_index[1] == i
                atoms = bond_data.hyperedge_index[0][mask].tolist()
                
                # 获取对应的特征
                if hasattr(bond_data, 'hyperedge_attr') and i < bond_data.hyperedge_attr.shape[0]:
                    features = bond_data.hyperedge_attr[i]
                else:
                    # 如果没有特征，创建默认的bond特征
                    features = torch.zeros(19, dtype=torch.float32)  # bonds的标准特征维度
                
                hyperedge_data.append({
                    'atoms': atoms,
                    'features': features,
                    'type': 'bond'
                })
        
        return hyperedge_data
    
    def _compute_hyperedges_by_type(self, mol: Chem.Mol, edge_type: str, 
                                   global_stats: Dict) -> List[Dict]:
        """计算特定类型的超边（复用现有逻辑）"""
        hyperedge_data = []
        seen_hyperedges = set()
        
        if edge_type == 'ring':
            # 复用现有的ring构建逻辑
            try:
                sssr = Chem.GetSymmSSSR(mol)
                for ring in sssr:
                    if len(ring) >= 3:
                        ring_atoms = tuple(sorted(ring))
                        if ring_atoms not in seen_hyperedges:
                            seen_hyperedges.add(ring_atoms)
                            ring_features = get_ring_features(list(ring), mol)
                            hyperedge_data.append({
                                'atoms': list(ring_atoms),
                                'features': ring_features,
                                'type': 'ring'
                            })
            except Exception as e:
                logger.warning(f"Ring identification failed: {e}")
        
        elif edge_type == 'functional_group':
            # 复用现有的functional group识别逻辑
            try:
                fg_results = identify_all_functional_groups(mol, self.config)
                for fg_atoms, fg_type, fg_features in fg_results:
                    if len(fg_atoms) >= 2:
                        fg_tuple = tuple(fg_atoms)  # 保持原有顺序
                        if fg_tuple not in seen_hyperedges:
                            seen_hyperedges.add(fg_tuple)
                            hyperedge_data.append({
                                'atoms': list(fg_tuple),
                                'features': fg_features,
                                'type': 'functional_group'
                            })
            except Exception as e:
                logger.warning(f"Functional group identification failed: {e}")
        
        elif edge_type == 'hydrogen_bond':
            # 复用现有的氢键识别逻辑
            try:
                h_bond_identifier = HydrogenBondIdentifier(self.config)
                h_bonds = h_bond_identifier.identify(mol)
                
                for h_bond_atoms, h_bond_attrs in h_bonds:
                    if len(h_bond_atoms) == 3:
                        h_bond_tuple = tuple(h_bond_atoms)  # 保持Donor-H-Acceptor顺序
                        if h_bond_tuple not in seen_hyperedges:
                            seen_hyperedges.add(h_bond_tuple)
                            hyperedge_data.append({
                                'atoms': list(h_bond_tuple),
                                'features': h_bond_attrs,
                                'type': 'hydrogen_bond'
                            })
            except Exception as e:
                logger.warning(f"Hydrogen bond identification failed: {e}")
        
        elif edge_type == 'conjugated_system':
            # 复用现有的共轭系统识别逻辑
            try:
                from .functional_groups import ConjugatedSystemIdentifier
                conj_identifier = ConjugatedSystemIdentifier()
                conjugated_systems = conj_identifier.identify(mol)
                
                for system in conjugated_systems:
                    if len(system) >= 3:
                        system_tuple = tuple(system)  # 保持原有结构
                        if system_tuple not in seen_hyperedges:
                            seen_hyperedges.add(system_tuple)
                            conj_features = conj_identifier.extract_features(system, mol)
                            hyperedge_data.append({
                                'atoms': list(system_tuple),
                                'features': conj_features,
                                'type': 'conjugated_system'
                            })
            except Exception as e:
                logger.warning(f"Conjugated system identification failed: {e}")
        
        return hyperedge_data
    
    def _reassemble_data_object(self, original_data: Data, 
                               all_hyperedge_data: List[Dict]) -> Data:
        """重新组装Data对象，合并所有超边"""
        try:
            # 1. 使用现有的维度统一逻辑
            all_hyperedge_data = ensure_canonical_feature_dims(all_hyperedge_data, self.config)
            unified_dim = compute_unified_hyperedge_dim(all_hyperedge_data, self.config)
            padded_features = pad_to_unified_dim(all_hyperedge_data, unified_dim)
            
            # 2. 构建新的hyperedge_index和hyperedge_attr
            hyperedge_list = [item['atoms'] for item in all_hyperedge_data]
            hyperedge_attr = torch.stack(padded_features) if padded_features else torch.zeros((0, unified_dim))

            # 2.1 每条超边的类型ID（便于下游统计/解释）
            type_names = [str(item.get('type', 'unknown')) for item in all_hyperedge_data]
            unique_types = sorted(set(type_names)) if type_names else ['unknown']
            type_to_id = {t: i for i, t in enumerate(unique_types)}
            hyperedge_type = torch.tensor([type_to_id[t] for t in type_names], dtype=torch.long)
            
            # 3. 创建hyperedge_index
            hyperedge_index = []
            for i, atoms in enumerate(hyperedge_list):
                for atom in atoms:
                    hyperedge_index.append([atom, i])
            
            hyperedge_index = torch.tensor(hyperedge_index).t().contiguous() if hyperedge_index else torch.empty((2, 0), dtype=torch.long)
            
            # 4. 创建必需的mask
            num_nodes = original_data.num_nodes
            num_hyperedges = hyperedge_attr.shape[0]
            
            # 创建默认的masks（训练时会动态更新）
            node_mask = torch.zeros(num_nodes, dtype=torch.bool)
            edge_mask = torch.zeros(num_hyperedges, dtype=torch.bool)
            
            # 5. 创建新的Data对象
            enhanced_data = Data(
                x=original_data.x,  # 复用原始节点特征
                hyperedge_index=hyperedge_index,
                hyperedge_attr=hyperedge_attr,
                hyperedge_type=hyperedge_type,
                node_mask=node_mask,  # 添加node_mask
                edge_mask=edge_mask,  # 添加edge_mask
                id=original_data.id,
                smiles=original_data.smiles,
                # 保留关键的标签信息，确保下游评估正常工作
                y=original_data.y,  # 标签值（必需）
                y_mask=getattr(original_data, 'y_mask', None),  # 标签掩码（可选）
                split=getattr(original_data, 'split', None),    # 数据集分割（可选）
                num_nodes=num_nodes,
                hyperedge_dim=unified_dim,
                num_hyperedges=num_hyperedges,
                meta_info={
                    'hyperedge_types': list(set(type_names)),
                    'hyperedge_type_map': {int(i): t for t, i in type_to_id.items()},
                    'incremental_build': True,
                    'original_bonds_only': True
                }
            )

            # Runtime attach a trivial RDKit atom index mapping for nodes (KISS):
            # Assumes nodes correspond 1:1 to RDKit atoms in order used during preprocessing.
            try:
                enhanced_data.rdkit_idx = torch.arange(num_nodes, dtype=torch.long)
            except Exception:
                pass
            
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Error reassembling data object: {e}")
            raise
    
    def _make_cache_key(self) -> str:
        """生成包含数据集名称和超边类型的缓存键（支持分片）"""
        base_key = self._make_base_cache_key()
        
        if self.is_sharded:
            return f"{base_key}_shard_{self.shard_index}_of_{self.total_shards}"
        else:
            return base_key
    
    def _make_base_cache_key(self) -> str:
        """生成基础缓存键（不包含分片信息）"""
        dataset_name = self.bonds_data_dir.name  # 如：zinc_druglikehypergraph
        hyperedge_types = "_".join(sorted(self.target_types))  # 如：bond_ring
        simple_hash = hashlib.md5(str(sorted(self.target_types)).encode()).hexdigest()[:6]
        return f"{dataset_name}_{hyperedge_types}_{simple_hash}"
    
    def _load_bonds_data(self) -> List[Data]:
        """加载bonds-only数据（支持分片加载）"""
        all_graphs = []
        batch_files = sorted(self.bonds_data_dir.glob("batch_*.pt"))
        
        if not batch_files:
            raise FileNotFoundError(f"No batch files found in {self.bonds_data_dir}")
        
        # 如果是分片模式，只加载对应的batch文件
        if self.is_sharded:
            # 按分片索引筛选batch文件
            shard_files = [batch_files[i] for i in range(self.shard_index, len(batch_files), self.total_shards)]
            batch_files = shard_files
            logger.info(f"Shard {self.shard_index}: processing {len(shard_files)} batch files")
        
        desc = f"Loading bonds data (shard {self.shard_index})" if self.is_sharded else "Loading bonds data"
        
        for batch_file in tqdm(batch_files, desc=desc):
            try:
                with open(batch_file, 'rb') as f:
                    graphs = torch.load(f, weights_only=False)
                all_graphs.extend(graphs)
            except Exception as e:
                logger.warning(f"Error loading {batch_file}: {e}")
        
        return all_graphs
    
    def _load_global_stats(self) -> Dict:
        """加载或计算全局统计信息"""
        stats_file = self.bonds_data_dir / "global_stats.pkl"
        if stats_file.exists():
            # 使用现有的统计文件
            with open(stats_file, 'rb') as f:
                return pickle.load(f)
        else:
            # 动态计算统计信息，复用现有轮子
            logger.info("Global stats file not found, computing from bonds-only data...")
            return self._compute_stats_from_bonds_data(stats_file)
    
    def _try_merge_shards(self) -> Optional[List[Data]]:
        """尝试从已有的分片缓存合并数据集"""
        base_key = self._make_base_cache_key()
        
        # 自动检测可用的分片数量
        detected_shards = self._detect_available_shards(base_key)
        
        if not detected_shards:
            logger.info("No shard cache files found")
            return None
        
        logger.info(f"Found {len(detected_shards)} shard cache files, attempting to merge...")
        
        all_graphs = []
        total_graphs = 0
        
        for shard_info in detected_shards:
            shard_file = shard_info['file']
            shard_index = shard_info['index']
            
            try:
                logger.info(f"Loading shard {shard_index}: {shard_file}")
                with open(shard_file, 'rb') as f:
                    graphs = torch.load(f, weights_only=False)
                all_graphs.extend(graphs)
                total_graphs += len(graphs)
                logger.info(f"Shard {shard_index}: loaded {len(graphs)} graphs")
            except Exception as e:
                logger.error(f"Failed to load shard {shard_index}: {e}")
                return None
        
        logger.info(f"Successfully merged {total_graphs} graphs from {len(detected_shards)} shards")
        return all_graphs
    
    def _detect_available_shards(self, base_key: str) -> List[Dict]:
        """检测可用的分片缓存文件"""
        pattern = f"incremental_{base_key}_shard_*_of_*.pt"
        shard_files = list(self.cache_dir.glob(pattern))
        
        detected_shards = []
        
        for shard_file in shard_files:
            # 解析文件名获取分片信息
            filename = shard_file.name
            try:
                # 匹配格式：incremental_{base_key}_shard_{index}_of_{total}.pt
                parts = filename.replace(f"incremental_{base_key}_shard_", "").replace(".pt", "")
                index_part, total_part = parts.split("_of_")
                shard_index = int(index_part)
                total_shards = int(total_part)
                
                detected_shards.append({
                    'file': shard_file,
                    'index': shard_index,
                    'total': total_shards
                })
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse shard filename {filename}: {e}")
                continue
        
        # 按shard_index排序
        detected_shards.sort(key=lambda x: x['index'])
        
        return detected_shards
    
    def check_shards_complete(self, total_shards: int) -> bool:
        """检查指定数量的分片是否都已完成"""
        base_key = self._make_base_cache_key()
        
        for shard_idx in range(total_shards):
            shard_cache_file = self.cache_dir / f"incremental_{base_key}_shard_{shard_idx}_of_{total_shards}.pt"
            if not shard_cache_file.exists():
                return False
        
        return True
    
    def get_shard_status(self, total_shards: int) -> Dict:
        """获取分片处理状态"""
        base_key = self._make_base_cache_key()
        status = {
            'total_shards': total_shards,
            'completed_shards': 0,
            'missing_shards': [],
            'completed_files': []
        }
        
        for shard_idx in range(total_shards):
            shard_cache_file = self.cache_dir / f"incremental_{base_key}_shard_{shard_idx}_of_{total_shards}.pt"
            if shard_cache_file.exists():
                status['completed_shards'] += 1
                status['completed_files'].append(str(shard_cache_file))
            else:
                status['missing_shards'].append(shard_idx)
        
        return status
    
    def _compute_stats_from_bonds_data(self, stats_file: Path) -> Dict:
        """
        从bonds-only数据计算全局统计信息，复用现有轮子
        
        Args:
            stats_file: 统计文件保存路径
        
        Returns:
            全局统计信息字典
        """
        try:
            import pandas as pd
            from .molecule_features import get_global_feature_statistics
            
            # 1. 收集bonds-only数据中的SMILES
            logger.info("Collecting SMILES from bonds-only data...")
            smiles_list = []
            
            # 遍历bonds数据目录下的batch文件
            batch_files = sorted(self.bonds_data_dir.glob("batch_*.pt"))
            if not batch_files:
                raise FileNotFoundError(f"No batch files found in {self.bonds_data_dir}")
            
            # 从batch文件中提取SMILES
            for batch_file in batch_files[:3]:  # 限制样本数量以提高效率
                try:
                    batch_data = torch.load(batch_file, map_location="cpu", weights_only=False)
                    if isinstance(batch_data, list):
                        for data in batch_data:
                            if hasattr(data, 'smiles') and data.smiles:
                                smiles_list.append(data.smiles)
                    logger.debug(f"Extracted {len(smiles_list)} SMILES so far from {batch_file.name}")
                except Exception as e:
                    logger.warning(f"Could not extract SMILES from {batch_file}: {e}")
                    continue
            
            if not smiles_list:
                raise ValueError("No SMILES found in bonds-only data")
            
            logger.info(f"Collected {len(smiles_list)} SMILES for statistics computation")
            
            # 2. 转换为RDKit分子对象
            logger.info("Converting SMILES to molecules...")
            molecules = []
            for smiles in tqdm(smiles_list, desc="Converting molecules"):
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    molecules.append(mol)
            
            if not molecules:
                raise ValueError("No valid molecules found for statistics computation")
            
            logger.info(f"Successfully converted {len(molecules)} molecules")
            
            # 3. 复用现有轮子计算统计信息
            logger.info("Computing global feature statistics...")
            global_stats = get_global_feature_statistics(molecules)
            
            # 4. 保存统计信息供后续使用
            with open(stats_file, 'wb') as f:
                pickle.dump(global_stats, f)
            logger.info(f"Saved computed global statistics to {stats_file}")
            
            return global_stats
            
        except Exception as e:
            logger.error(f"Failed to compute global stats from bonds data: {e}")
            logger.info("Using fallback default statistics...")
            # 返回一个基本的默认统计信息
            return self._get_default_global_stats()
    
    def _get_default_global_stats(self) -> Dict:
        """
        提供默认的全局统计信息作为备用方案
        """
        return {
            'cont_mean': [6.0, 0.0, 1.0],  # atomic_num, formal_charge, total_num_hs
            'cont_std': [3.0, 1.0, 1.0],
            'bond_cont_means': [2.0],  # avg_hybridization
            'bond_cont_stds': [1.0],
            'degree_cat': list(range(7)),  # degrees 0-6
            'hybridization_cats': [
                Chem.rdchem.HybridizationType.UNSPECIFIED,
                Chem.rdchem.HybridizationType.S,
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2
            ],
            'atom_chiral_cats': [
                Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                Chem.rdchem.ChiralType.CHI_OTHER
            ],
            'bond_stereo_cats': [
                Chem.rdchem.BondStereo.STEREONONE,
                Chem.rdchem.BondStereo.STEREOZ,
                Chem.rdchem.BondStereo.STEREOE,
                Chem.rdchem.BondStereo.STEREOCIS,
                Chem.rdchem.BondStereo.STEREOTRANS,
                Chem.rdchem.BondStereo.STEREOANY
            ],
            'bond_type_cats': [
                Chem.rdchem.BondType.SINGLE,
                Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE,
                Chem.rdchem.BondType.AROMATIC
            ]
        }
