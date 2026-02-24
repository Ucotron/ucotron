"use client";

import { useEffect, useRef, useCallback } from "react";
import * as d3 from "d3";
import type { GraphNode, GraphEdge } from "@/lib/api";

/** Node type → color mapping. */
const NODE_COLORS: Record<string, string> = {
  Entity: "#00F0FF",  // cyan (primary)
  Event: "#00B8CC",   // cyan hover
  Fact: "#00FF94",    // success green
  Skill: "#FFB800",   // warning amber
};

const DEFAULT_COLOR = "#94A3B8"; // slate

interface SimNode extends d3.SimulationNodeDatum {
  id: number;
  content: string;
  node_type: string;
  timestamp: number;
  community_id: number | null;
}

interface SimLink extends d3.SimulationLinkDatum<SimNode> {
  weight: number;
}

interface ForceGraphProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  selectedNodeId: number | null;
  onSelectNode: (id: number | null) => void;
  highlightedNodeIds?: Set<number> | null;
}

export function ForceGraph({ nodes, edges, selectedNodeId, onSelectNode, highlightedNodeIds }: ForceGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const simulationRef = useRef<d3.Simulation<SimNode, SimLink> | null>(null);

  const handleClick = useCallback(
    (id: number | null) => {
      onSelectNode(id);
    },
    [onSelectNode]
  );

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;

    const container = svg.parentElement;
    if (!container) return;

    const width = container.clientWidth;
    const height = container.clientHeight;

    // Clear previous render.
    d3.select(svg).selectAll("*").remove();

    if (nodes.length === 0) return;

    // Build simulation data (deep clone to avoid mutating props).
    const simNodes: SimNode[] = nodes.map((n) => ({ ...n }));
    const nodeMap = new Map(simNodes.map((n) => [n.id, n]));

    const simLinks: SimLink[] = edges
      .filter((e) => nodeMap.has(e.source) && nodeMap.has(e.target))
      .map((e) => ({
        source: nodeMap.get(e.source)!,
        target: nodeMap.get(e.target)!,
        weight: e.weight,
      }));

    const svgSel = d3
      .select(svg)
      .attr("width", width)
      .attr("height", height)
      .attr("viewBox", [0, 0, width, height]);

    // Root group for zoom/pan.
    const g = svgSel.append("g");

    // Zoom behavior.
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 8])
      .on("zoom", (event) => {
        g.attr("transform", event.transform);
      });

    svgSel.call(zoom);

    // Click on background deselects.
    svgSel.on("click", (event) => {
      if (event.target === svg) {
        handleClick(null);
      }
    });

    // Edges (links).
    const link = g
      .append("g")
      .attr("class", "links")
      .selectAll<SVGLineElement, SimLink>("line")
      .data(simLinks)
      .join("line")
      .attr("stroke", "rgba(0, 240, 255, 0.25)")
      .attr("stroke-opacity", 0.4)
      .attr("stroke-width", (d) => Math.max(0.5, d.weight * 2));

    // Node groups.
    const node = g
      .append("g")
      .attr("class", "nodes")
      .selectAll<SVGGElement, SimNode>("g")
      .data(simNodes)
      .join("g")
      .attr("cursor", "pointer")
      .on("click", (event, d) => {
        event.stopPropagation();
        handleClick(d.id);
      })
      .call(
        d3
          .drag<SVGGElement, SimNode>()
          .on("start", (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on("drag", (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on("end", (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          })
      );

    // Degree-based radius.
    const degreeMap = new Map<number, number>();
    for (const e of edges) {
      degreeMap.set(e.source, (degreeMap.get(e.source) || 0) + 1);
      degreeMap.set(e.target, (degreeMap.get(e.target) || 0) + 1);
    }

    const radiusScale = d3.scaleSqrt().domain([0, d3.max([...degreeMap.values()]) || 1]).range([4, 16]);

    // Circle for each node.
    node
      .append("circle")
      .attr("r", (d) => radiusScale(degreeMap.get(d.id) || 0))
      .attr("fill", (d) => NODE_COLORS[d.node_type] || DEFAULT_COLOR)
      .attr("stroke", (d) => {
        if (d.id === selectedNodeId) return "#00F0FF";
        if (highlightedNodeIds && highlightedNodeIds.has(d.id)) return "#FFB800";
        return "#0F1C2E";
      })
      .attr("stroke-width", (d) => {
        if (d.id === selectedNodeId || (highlightedNodeIds && highlightedNodeIds.has(d.id))) return 3;
        return 1.5;
      });

    // Label (truncated content).
    node
      .append("text")
      .text((d) => (d.content.length > 20 ? d.content.slice(0, 20) + "..." : d.content))
      .attr("font-size", "9px")
      .attr("dx", (d) => radiusScale(degreeMap.get(d.id) || 0) + 3)
      .attr("dy", "0.35em")
      .attr("fill", "currentColor")
      .attr("pointer-events", "none");

    // Hover tooltip.
    node
      .append("title")
      .text((d) => `${d.content}\nType: ${d.node_type}\nID: ${d.id}`);

    // Force simulation.
    const simulation = d3
      .forceSimulation(simNodes)
      .force("link", d3.forceLink<SimNode, SimLink>(simLinks).id((d) => d.id).distance(60).strength(0.4))
      .force("charge", d3.forceManyBody().strength(-120))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collide", d3.forceCollide().radius((d) => radiusScale(degreeMap.get((d as SimNode).id) || 0) + 2))
      .on("tick", () => {
        link
          .attr("x1", (d) => (d.source as SimNode).x!)
          .attr("y1", (d) => (d.source as SimNode).y!)
          .attr("x2", (d) => (d.target as SimNode).x!)
          .attr("y2", (d) => (d.target as SimNode).y!);

        node.attr("transform", (d) => `translate(${d.x},${d.y})`);
      });

    simulationRef.current = simulation;

    return () => {
      simulation.stop();
    };
  }, [nodes, edges, handleClick, highlightedNodeIds, selectedNodeId]);

  // Separate effect for highlighting selected node and search matches.
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const svgSel = d3.select(svg);
    svgSel.selectAll<SVGGElement, SimNode>(".nodes g")
      .select("circle")
      .attr("stroke", (d) => {
        if (d.id === selectedNodeId) return "#00F0FF";
        if (highlightedNodeIds && highlightedNodeIds.has(d.id)) return "#FFB800";
        return "#0F1C2E";
      })
      .attr("stroke-width", (d) => {
        if (d.id === selectedNodeId || (highlightedNodeIds && highlightedNodeIds.has(d.id))) return 3;
        return 1.5;
      });
  }, [selectedNodeId, highlightedNodeIds]);

  return (
    <div className="relative h-full w-full overflow-hidden rounded-lg border border-border bg-card">
      <svg ref={svgRef} className="h-full w-full" />
    </div>
  );
}
