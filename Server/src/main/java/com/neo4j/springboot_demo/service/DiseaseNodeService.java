package com.neo4j.springboot_demo.service;

import com.neo4j.springboot_demo.entity.nodes.DiseaseNode;
import com.neo4j.springboot_demo.entity.nodes.GeneNode;
import com.neo4j.springboot_demo.entity.nodes.TissueNode;

import java.util.List;
import java.util.Optional;

public interface DiseaseNodeService {

    // 创建节点
    void saveDiseaseNodeByDiseaseName(String diseaseName);

    // 删除节点
    void deleteById(Long id);

    // 获取所有节点
    List<DiseaseNode> listDiseaseNodes();

    // 根据id获得节点
    Optional<DiseaseNode> getDiseaseNodeById(Long id);

    // 根据diseaseName获取节点
    Optional<DiseaseNode> getDiseaseNodeByDiseaseName(String diseaseName);

    // 获取与基因间关系
    List<DiseaseNode> getRelationWithGeneNode(String geneName);

    // 保存与基因间关系
    void saveRelationWithGeneNode(String diseaseName, String geneName);

    // 获取与器官组织间关系
    List<DiseaseNode> getRelationWithTissueNode(String tissueName);

    // 保存与器官组织间关系
    void saveRelationWithTissueNode(String diseaseName, String tissueName);

    // 获取与疾病间关系
    List<DiseaseNode> getRelationWithDiseaseNode(String diseaseName);

    //保存与疾病间关系
    void saveRelationWithDiseaseNode(String firstDiseaseName, String secondDiseaseName);

}
