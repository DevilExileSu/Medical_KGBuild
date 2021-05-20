package com.neo4j.springboot_demo.service;

import com.neo4j.springboot_demo.entity.nodes.TissueNode;

import java.util.List;
import java.util.Optional;

public interface TissueNodeService {

    // 创建节点
    void saveTissueNodeByTissueName(String tissueName);

    // 删除节点
    void deleteById(Long id);

    // 获取所有节点
    List<TissueNode> listTissueNodes();

    // 根据id获得节点
    Optional<TissueNode> getTissueNodeById(Long id);

    // 根据diseaseName获取节点
    Optional<TissueNode> getTissueNodeByTissueName(String tissueName);

    // 获取与器官组织之间的关系
    List<TissueNode> getRelationWithTissueNode(String tissueName);

    // 保存与器官组织间的关系
    void saveRelationWithTissueNode(String firstTissueName, String secondTissueName);

    // 获取与疾病间关系
    List<TissueNode> getRelationWithDiseaseNode(String diseaseName);

}
