package com.neo4j.springboot_demo.service;

import com.neo4j.springboot_demo.entity.nodes.GeneNode;

import java.util.List;
import java.util.Optional;

public interface GeneNodeService {

    // 创建节点
    void saveGeneNodeByGeneName(String geneName);

    // 删除节点
    void deleteById(Long id);

    // 获取所有节点
    List<GeneNode> listGeneNodes();

    // 根据id获得节点
    Optional<GeneNode> getGeneNodeById(Long id);

    // 根据diseaseName获取节点
    Optional<GeneNode> getGeneNodeByGeneName(String geneName);

    // 获取与疾病间关系
    List<GeneNode> getRelationWithDiseaseNode(String diseaseName);

}
