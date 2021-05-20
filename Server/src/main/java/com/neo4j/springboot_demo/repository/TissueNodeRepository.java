package com.neo4j.springboot_demo.repository;

import com.neo4j.springboot_demo.entity.nodes.TissueNode;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.data.neo4j.repository.query.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;

public interface TissueNodeRepository extends Neo4jRepository<TissueNode, Long> {

    @Query("MATCH (tissue: TissueNode) RETURN tissue LIMIT 5")
    List<TissueNode> listTissueNodes();

    @Query("CREATE (tissue: TissueNode{tissueName:{tissueName}})")
    void saveTissueNode(@Param("tissueName") String tissueName);

    @Query("MATCH (tissue:TissueNode{tissueName:$tissueName}) RETURN tissue")
    TissueNode getTissueNodeByTissueName(@Param("tissueName") String tissueName);

    @Query("MATCH (tissue1:TissueNode{tissueName:$firstTissueName}), (tissue2:TissueNode{tissueName:$secondTissueName}) " +
            "MERGE (tissue1)-[:tissue_associated_with_tissue]->(tissue2)")
    void saveRelationWithTissueNode(String firstTissueName, String secondTissueName);

    @Query("MATCH (tissue1:TissueNode{tissueName:$tissueName})-[:tissue_associated_with_tissue]->(tissue2:TissueNode) RETURN tissue2")
    List<TissueNode> getRelationWithTissueNode(String tissueName);

    @Query("MATCH (a:DiseaseNode{diseaseName:$diseaseName})-[:disease_associated_with_tissue]->(b:TissueNode) RETURN b")
    List<TissueNode> getRelationWithDiseaseNode(String diseaseName);
}
