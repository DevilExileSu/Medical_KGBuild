<template>
  	<div class="canvas">
     <div :id="'g'+ g_id"  class="graphcontainer"></div>
        <div class="svg-set-box"></div>
	</div>
</template>

<script>
import * as d3 from 'd3';
export default {
	props: {
		keys:{
          type:String,
      },
	  data: {
		  type:Object,
	  },
	  keyWord: {
		  type: Array
	  }
	},
    data() {
    return {
		svg: null,
		node: this.data.entity,
		link: this.data.relation,
        color_map: {"disease":'rgba(255,0,0,0.5)', "gene":'rgba(0,0,255,0.5)', "tissue": 'rgba(255,255,0, 0.6)'},
		stroke_map:{"disease":'rgba(255,0,0, 0.8)', "gene":'rgba(0,0,255, 0.8)', "tissue": 'rgba(200,255,0, 0.8)'},
		node_size: {"disease": 25, "gene": 25, "tissue": 25},
        graph: {
	            nodes: [],
	            links: []
	    },
		g_id: this.keys,
		kw: [],
	
    };
    },
	watch: {
		keys: 'initgraph'
	},
    mounted() {

        this.initgraph();
     },
    methods: {
		addmaker() {
	            var arrowMarker = this.svg.append("marker")
	                .attr("id", "arrow")
					.attr("viewBox", "0 -5 10 10")
					.attr("refX", 20)
					.attr("refY", -0.5)
					.attr("markerWidth", 4)
					.attr("markerHeight", 4)
					.attr("orient", "auto")
	            var arrow_path = "M0,-5L10,0L0,5";// 定义箭头形状
	            arrowMarker.append("path").attr("d", arrow_path).attr("fill", "rgba(36,227,36, 0.7)");
	    },
		initgraph() {
			this.graph.nodes.length = 0
			// d3.selectAll("svg").remove();
			this.node = this.data.entity
			this.link = this.data.relation
			this.g_id = this.keys
			this.kw = this.keyWord!=null?this.keyWord:this.kw

			for(const item of this.link) {
				var sourceNode = this.node.filter(function (n) {
	                    return n.id === item.source;
	                })[0];
	            if (typeof(sourceNode) == 'undefined') return;
	            var targetNode = this.node.filter(function (n) {
	                    return n.id === item.target;
	                })[0];
	            if (typeof(targetNode) == 'undefined') return;

				this.graph.links.push({
					source: sourceNode.id,
					target: targetNode.id,
					relationship: item.relationship,
					width: 5,
					color: 'rgba(36,227,36, 0.4)',
					id: item.id
				})
			}


			for(const item of this.node) {
				this.graph.nodes.push({ 
					name: item.name,
					id: item.id, 
					type: item.type,
					size: this.node_size[item.type],
					color: this.color_map[item.type],
					keyWord: this.kw.indexOf(item.name) != -1
					})

				// 返回数据改为nodes(name, pos, type, id), relations(source_id, target_id, name), 
				// 		this.graph.nodes = this.test.;
				// this.graph.links = jsondata.relationship;
				}
			// console.log(this.graph.nodes)

			var graph_id = "#g".concat(this.g_id);
			var graphcontainer = d3.select(graph_id);
			
			var width = window.screen.width;
	        var height = window.screen.height - 354;//
	        this.svg = graphcontainer.append("svg");
	        this.svg.attr("width", width);
	        this.svg.attr("height", height);
		//利用d3.forceSimulation()定义关系图 包括设置边link、排斥电荷charge、关系图中心点
		var simulation = d3.forceSimulation()
	        .force("link", d3.forceLink()
			.distance(function(d){
					return d.relationship.length * 5;
						})
			.id(d => d.id))
	        .force("charge", d3.forceManyBody().strength(-150))
	        .force("center", d3.forceCenter(width / 2, height / 2));
	    	//g用于绘制所有边,selectALL选中所有的line,并绑定数据data(graph.links),enter().append("line")添加元素
	    	//数据驱动文档,设置边的粗细

			this.addmaker()

	    	var link = this.svg.append("g").attr("class", "line").attr("stroke-opacity", 0.6).attr("fill", "none").selectAll("path").data(this.graph.links)
	    	.join("path").attr("stroke", function(d) {
				return d.color;		
			}).attr("stroke-width", function(d) {
	    		//return Math.sqrt(d.value);
	    		return d.width; //所有线宽度均为1
	    	}).attr("marker-end", "url(#arrow)")
			.attr("id", function (d) {
	                    return "invis_" + d.source + "-" + d.relationship + "-" + d.target;
	                })

			var linkText = this.svg.append("g").attr("class", "line-text").selectAll("line-text").data(this.graph.links).enter()
			 .append("text") 
			  .style('fill', '#3f3f3f')
	                .append("textPath")
	                .attr("startOffset", "50%")
	                .attr("text-anchor", "middle")
	                .attr("xlink:href", function(d) {
	                    return "#invis_" + d.source + "-" + d.relationship + "-" + d.target;
	                })
	                .style("font-size", 6)
			.text(function(d) {
                return d.relationship;
            })
			
			//添加所有的点
	    	//selectAll("circle")选中所有的圆并绑定数据,圆的直径为d.size
	    	//再定义圆的填充色,同样数据驱动样式,圆没有描边,圆的名字为d.id
	    	//call()函数：拖动函数,当拖动开始绑定dragstarted函数，拖动进行和拖动结束也绑定函数
			var node = this.svg.append("g").attr("class", "nodes").selectAll("circle").data(this.graph.nodes).enter()
			.append("circle").attr("r", function(d) {
	    		return d.size;
	    	}).attr("fill", function(d) {
	    		return d.color;
	    	}).attr("stroke", d=>this.stroke_map[d.type])
      		.attr("stroke-width", 2)
			.call(d3.drag()
	    		.on("start", dragstarted)
	    		.on("drag", dragged)
	    		.on("end", dragended)
	    	);

			//显示所有的文本 
	    	var nodeText = this.svg.append("g").attr("class", "node-text").selectAll("node-text").data(this.graph.nodes).enter()
	    	.append("text").attr("font-size",function(d) {
				return d.keyWord?24:13
			}).attr("fill", '#3f3f3f').attr('name', function(d) {
                return d.name;
            }).text(function(d) {
                return d.name;
            }).attr('text-anchor', 'middle').call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended)
            );

			link.append("title").text(function(d) {
				return "relationship: " + d.relationship 
			})

			node.append("title").text(function(d) {
	    		return d.type + ": " + d.name;
	    	})
			
	    	simulation
	            .nodes(this.graph.nodes);

	        simulation.force("link")
	            .links(this.graph.links);
			
			simulation.on("tick", ticked)

			function linkArc(d) {
				const r = Math.hypot(d.target.x - d.source.x, d.target.y - d.source.y);
				return `
					M${d.source.x},${d.source.y}
					A${r},${r} 0 0,1 ${d.target.x},${d.target.y}
				`;
			}
			function ticked() {
	            link
	                .attr("d", linkArc)
	            node
	                .attr("cx", function(d) {
	                    return d.x;
	                })
	                .attr("cy", function(d) {
	                    return d.y;
	                });

	            nodeText
	            .attr('transform', function(d) {
	                return 'translate(' + d.x + ',' + (d.y + 8) + ')';
	            });

				linkText
				.attr('transform', function(d) {
	                return 'translate(' + (d.source.x + d.target.x)/2 + ',' + (d.source.y+d.target.y)/2 + ')';
	            });
	        }
			this.svg.call(d3.zoom().on("zoom", zoomed).filter(filter))

				function zoomed() {
					// if() {
						d3.selectAll('.nodes').attr("transform",d3.zoomTransform(this));
						d3.selectAll('.node-text').attr("transform",d3.zoomTransform(this));
						d3.selectAll('.line').attr("transform",d3.zoomTransform(this));
						d3.selectAll('.line-text').attr("transform",d3.zoomTransform(this));
					// }
				}
				function filter() {
					return window.event.shiftKey;
				}
			    //开始拖动并更新相应的点
				function dragstarted(event) {
					if (!event.active) simulation.alphaTarget(0.3).restart();
					event.subject.fx = event.subject.x;
					event.subject.fy = event.subject.y;
				}
				
				function dragged(event) {
					event.subject.fx = event.x;
					event.subject.fy = event.y;
				}
				
				function dragended(event) {
					if (!event.active) simulation.alphaTarget(0);
					event.subject.fx = null;
					event.subject.fy = null;
				}

		}
    },
}
</script>

<style>
.canvas{
	width:100%;
	height: 100%;
}
circle{
	cursor:pointer;
}
.node-text {
  font: Microsoft YaHei;
  pointer-events: none;
  user-select: none;
  word-wrap: break-word;
  word-break: normal;
}
.line-text {
  font: Microsoft YaHei;
  pointer-events: none;
  user-select: none;
  overflow: hidden;
  font-size:13px;
}


.graphcontainer{
	
 	background:#fff;
}
</style>