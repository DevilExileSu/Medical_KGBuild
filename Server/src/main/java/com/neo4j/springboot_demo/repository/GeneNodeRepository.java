package com.neo4j.springboot_demo.repository;

import com.neo4j.springboot_demo.entity.nodes.GeneNode;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.data.neo4j.repository.query.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;

public interface GeneNodeRepository extends Neo4jRepository<GeneNode, Long> {
    @Query("MATCH (gene:GeneNode) RETURN gene LIMIT 5")
    List<GeneNode> listGeneNodes();

    @Query("CREATE (gene:GeneNode{geneName:{geneName}})")
    void saveGeneNode(@Param("geneName") String geneName);

    @Query("MATCH (gene:GeneNode{geneName:$geneName}) RETURN gene")
    GeneNode getGeneNodeByGeneName(@Param("geneName") String geneName);

    @Query("MATCH  (a:GeneNode)-[:gene_associated_with_disease]->(b:DiseaseNode{diseaseName:$diseaseName}) RETURN a")
    List<GeneNode> getRelationWithDiseaseNode(String diseaseName);
}
