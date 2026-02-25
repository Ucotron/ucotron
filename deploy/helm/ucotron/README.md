# Ucotron Helm Chart

Helm chart for deploying Ucotron (cognitive trust infrastructure for AI) on Kubernetes.

## Prerequisites

- Kubernetes 1.24+
- Helm 3.x
- PV provisioner support (for persistent storage)

## Installation

### Quick Start

```bash
helm install ucotron ./deploy/helm/ucotron
```

### With Custom Values

```bash
helm install ucotron ./deploy/helm/ucotron \
  --set config.server.workers=8 \
  --set persistence.size=50Gi \
  --set resources.limits.memory=8Gi
```

### From Custom Image

```bash
helm install ucotron ./deploy/helm/ucotron \
  --set image.repository=ghcr.io/ucotron/ucotron/ucotron-server \
  --set image.tag=v0.1.0
```

### With Ingress (nginx)

```bash
helm install ucotron ./deploy/helm/ucotron \
  --set ingress.enabled=true \
  --set ingress.className=nginx \
  --set "ingress.hosts[0].host=ucotron.example.com" \
  --set "ingress.hosts[0].paths[0].path=/" \
  --set "ingress.hosts[0].paths[0].pathType=Prefix"
```

## Uninstalling

```bash
helm uninstall ucotron
```

Note: The PVC for LMDB data is **not** deleted on uninstall to prevent data loss. Delete it manually if needed:

```bash
kubectl delete pvc ucotron-data
```

## Configuration

See `values.yaml` for the full list of configurable parameters.

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `replicaCount` | Number of replicas | `1` |
| `image.repository` | Container image | `ghcr.io/ucotron/ucotron/ucotron-server` |
| `image.tag` | Image tag | Chart appVersion |
| `service.type` | Kubernetes service type | `ClusterIP` |
| `service.port` | Service port | `8420` |
| `persistence.enabled` | Enable persistent storage | `true` |
| `persistence.size` | PVC size | `10Gi` |
| `config.server.workers` | Worker threads | `4` |
| `config.server.logLevel` | Log level | `info` |
| `config.storage.mode` | Storage mode | `embedded` |
| `config.models.modelsDir` | Models directory | `/app/models` |
| `ingress.enabled` | Enable Ingress | `false` |
| `resources.requests.cpu` | CPU request | `500m` |
| `resources.requests.memory` | Memory request | `1Gi` |
| `resources.limits.cpu` | CPU limit | `2` |
| `resources.limits.memory` | Memory limit | `4Gi` |

### Health Checks

- **Liveness**: TCP socket on port 8420 (fast, low overhead)
- **Readiness**: HTTP GET `/api/v1/health` (ensures API is responding)

### Persistence

LMDB data is stored in a PersistentVolumeClaim. The Deployment uses `strategy: Recreate` since LMDB does not support concurrent writers from multiple pods.

To use an existing PVC:

```bash
helm install ucotron ./deploy/helm/ucotron \
  --set persistence.existingClaim=my-existing-pvc
```

To disable persistence (ephemeral, data lost on restart):

```bash
helm install ucotron ./deploy/helm/ucotron \
  --set persistence.enabled=false
```

## Testing

Run the built-in Helm test:

```bash
helm test ucotron
```

Validate templates without installing:

```bash
helm template ucotron ./deploy/helm/ucotron
helm lint ./deploy/helm/ucotron
```
