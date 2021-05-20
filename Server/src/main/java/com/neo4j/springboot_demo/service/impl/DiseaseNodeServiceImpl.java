package com.neo4j.springboot_demo.service.impl;

import com.neo4j.springboot_demo.entity.nodes.DiseaseNode;
import com.neo4j.springboot_demo.entity.nodes.GeneNode;
import com.neo4j.springboot_demo.entity.nodes.TissueNode;
import com.neo4j.springboot_demo.repository.DiseaseNodeRepository;
import com.neo4j.springboot_demo.service.DiseaseNodeService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class DiseaseNodeServiceImpl implements DiseaseNodeService {

    @Autowired
    private DiseaseNodeRepository diseaseRepository;

    @Override
    public void saveDiseaseNodeByDiseaseName(String diseaseName) {
        diseaseRepository.saveDiseaseNode(diseaseName);
    }

    @Override
    public void deleteById(Long id) {
        diseaseRepository.deleteById(id);
    }

    @Override
    public List<DiseaseNode> listDiseaseNodes() {
        return diseaseRepository.listDiseaseNodes();
    }

    @Override
    public Optional<DiseaseNode> getDiseaseNodeById(Long id) {
        return diseaseRepository.findById(id);
    }

    @Override
    public Optional<DiseaseNode> getDiseaseNodeByDiseaseName(String diseaseName) {
        DiseaseNode diseaseNode = diseaseRepository.getDiseaseNodeByDiseaseName(diseaseName);
        return Optional.ofNullable(diseaseNode);
    }

    @Override
    public List<DiseaseNode> getRelationWithGeneNode(String geneName) {
        return diseaseRepository.getRelationWithGeneNode(geneName);
    }

    @Override
    public void saveRelationWithGeneNode(String diseaseName, String geneName) {
        diseaseRepository.saveRelationWithGeneNode(diseaseName, geneName);
    }

    @Override
    public List<DiseaseNode> getRelationWithTissueNode(String tissueName) {
        return diseaseRepository.getRelationWithTissueNode(tissueName);
    }

    @Override
    public void saveRelationWithTissueNode(String diseaseName, String tissueName) {
        diseaseRepository.saveRelationWithTissueNode(diseaseName, tissueName);
    }

    @Override
    public List<DiseaseNode> getRelationWithDiseaseNode(String diseaseName) {
        return diseaseRepository.getRelationWithDiseaseNode(diseaseName);
    }

    @Override
    public void saveRelationWithDiseaseNode(String firstDiseaseName, String secondDiseaseName) {
        diseaseRepository.saveRelationWithDiseaseNode(firstDiseaseName, secondDiseaseName);
    }

}
