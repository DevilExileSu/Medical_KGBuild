package com.neo4j.springboot_demo.repository;

import com.neo4j.springboot_demo.entity.nodes.DiseaseNode;
import com.neo4j.springboot_demo.entity.nodes.GeneNode;
import com.neo4j.springboot_demo.entity.nodes.TissueNode;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.data.neo4j.repository.query.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;

public interface DiseaseNodeRepository extends Neo4jRepository<DiseaseNode, Long> {
    @Query("MATCH (disease:DiseaseNode) RETURN disease LIMIT 5")
    List<DiseaseNode> listDiseaseNodes();

    @Query("CREATE (disease:DiseaseNode{diseaseName:{diseaseName}})")
    void saveDiseaseNode(@Param("diseaseName") String diseaseName);

    @Query("MATCH (disease:DiseaseNode{diseaseName:$diseaseName}) RETURN disease")
    DiseaseNode getDiseaseNodeByDiseaseName(@Param("diseaseName") String diseaseName);


    @Query("MATCH (gene:GeneNode{geneName:$geneName}), (disease: DiseaseNode{diseaseName:$diseaseName}) " +
            "MERGE (gene)-[:gene_associated_with_disease]->(disease)")
    void saveRelationWithGeneNode(String diseaseName, String geneName);

    @Query("MATCH  (a:GeneNode{geneName:$geneName})-[:gene_associated_with_disease]->(b:DiseaseNode) RETURN b LIMIT 5")
    List<DiseaseNode> getRelationWithGeneNode(String geneName);

    @Query("MATCH (disease:DiseaseNode{diseaseName:$diseaseName}), (tissue:TissueNode{tissueName:$tissueName})" +
            "MERGE (disease)-[:disease_associated_with_tissue]->(tissue)")
    void saveRelationWithTissueNode(String diseaseName, String tissueName);

    @Query("MATCH (a:DiseaseNode)-[:disease_associated_with_tissue]->(b:TissueNode{tissueName:$tissueName}) RETURN a")
    List<DiseaseNode> getRelationWithTissueNode(String tissueName);

    @Query("MATCH (disease1:DiseaseNode{diseaseName:$firstDiseaseName}), (disease2:DiseaseNode{diseaseName:$secondDiseaseName}) " +
            "MERGE (disease1)-[:disease_associated_with_disease]->(disease2)")
    void saveRelationWithDiseaseNode(String firstDiseaseName, String secondDiseaseName);

    @Query("MATCH (disease1:DiseaseNode{diseaseName:$diseaseName})-[:disease_associated_with_disease]->(disease2:DiseaseNode) RETURN disease2 ")
    List<DiseaseNode> getRelationWithDiseaseNode(String diseaseName);
}
