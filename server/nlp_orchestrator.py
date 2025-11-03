def generate_playbook_for_cluster(cluster_id, sample_indices, indicators):
    return f"""
Playbook for cluster {cluster_id}
Indicators: {indicators}
Actions:
1. Block IPs in {indicators.get('ips', [])}
2. Investigate samples {sample_indices[:10]}
3. Update IDS signatures
"""
