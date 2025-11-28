# Artemis Tools Bundle

This bundle provides the following packages:

- [Artemis Runner](https://docs.artemis.turintech.ai/code-execution/custom-worker): Includes the `artemis-runner` script
- [Sourcekon](https://docs.artemis.turintech.ai/code-audit/profiling-with-sourcekon): Includes `sourcekon`,
  `sourcekon-stats`, `sourcekon-convert`, and `soureckon-git` scripts
- Artemis Client: Available as the `artemis_client` Python library

## Artemis Runner

### Installation

```sh
pip install --find-links wheels/ artemis-runner
```

### Configuration

#### [artemis.turintech.ai](https://artemis.turintech.ai)

When you first run `artemis-runner`, it will guide you through setting up your credentials. You'll be prompted for:

- Your Artemis email address
- Your Artemis password
- A unique name for your runner

The runner will validate your credentials and save them to a `.env.credentials` file in your working directory.

Alternatively, you can set these as environment variables:

```bash
export ARTEMIS_USERNAME="your@email.com"
export ARTEMIS_PASSWORD="your_password"
export ARTEMIS_RUNNER_NAME="your-runner-name"
```

#### On-premises Deployments

For on-premises deployments, you need to set the following in your environment variables or `.env.credentials`:

```dosini
ARTEMIS_ENVIRONMENT=custom
```

Alternatively, you can pass `--environment=custom` to the `artemis-runner` script.

Additionally, you need a `.env.custom` file in your working directory with the following template:

```dosini
#########  THANOS  ###########
THANOS_CLIENT_ID=
THANOS_CLIENT_SECRET=
THANOS_GRANT_TYPE=password
THANOS_HOST=
THANOS_PORT=80
THANOS_POSTFIX=/turintech-thanos/api
THANOS_HTTPS=False

#########  THOR  ###########
THOR_HTTPS=False
THOR_HOST=
THOR_PORT=80
THOR_POSTFIX=/turintech-thor/api

#########  FALCON  ###########
FALCON_HTTPS=False
FALCON_HOST=
FALCON_PORT=80
FALCON_POSTFIX=/turintech-falcon/api

#########  VISION  ###########
VISION_HTTPS=False
VISION_HOST=
VISION_PORT=80
VISION_POSTFIX=/turintech-vision/api

######### LOKI ##########
LOKI_HTTPS=False
LOKI_HOST=
LOKI_PORT=80
LOKI_POSTFIX=/turintech-loki
LOKI_READ_TIMEOUT=15
LOKI_CONNECT_TIMEOUT=10
```

Note:

- The `*_HOST` is typically the same for all components
- `THANOS_CLIENT_ID` and `THANOS_CLIENT_SECRET` can be obtained from your deployment configuration
- For HTTPS deployments, you may need to set `*_PORT=443` and `*_HTTPS=True`

### Basic Usage

When connecting to [artemis.turintech.ai](https://artemis.turintech.ai) (default configuration), simply start the
runner:

```sh
artemis-runner
```

On first run, the runner will:

1. Prompt you for your Artemis credentials
2. Validate the credentials with the Artemis platform
3. Save them to `.env.credentials` for future use
4. Start processing tasks

For on-premises deployments, ensure you have completed the configuration steps mentioned above before starting the
script.

Finally, make sure to set the runner name in your project build settings on the website to match your runner name so
tasks can be properly routed.

### SSL Verification Configuration

You can configure SSL certificate verification by setting the `ARTEMIS_SSL_VERIFY` environment variable:

```sh
export ARTEMIS_SSL_VERIFY=true  # Use system CA certificates
# or
export ARTEMIS_SSL_VERIFY=false  # Disable SSL verification (not recommended for production)
# or
export ARTEMIS_SSL_VERIFY=/path/to/your/ca/certificates.pem  # Use custom CA certificates
```

- `true` - Uses your system's CA certificates. If you don't have valid CAs installed on your machine, you may encounter
  errors.
- `false` - Disables certificate verification. This is not recommended for production environments due to security
  risks.
- `/path/to/certificates` - Allows you to specify a custom path to your CA certificates.

If you set this value to `true` and don't have valid CA certificates installed, you might encounter errors like:

```
[WARNING] 16:55:43,612 [client] (ThanosClient) Connection error, retrying: HTTPSConnectionPool(host='dev.artemis.turintech.ai', port=443): Max retries exceeded with url: /turintech-thanos/api/auth/login (Caused by SSLError(SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: Hostname mismatch, certificate is not valid for 'dev.artemis.turintech.ai'. (_ssl.c:1006)")))
```

In this case, you'll need to forcibly close the runner and either install proper CA certificates or use one of the other
options.

> **Warning**: All microservices used by the custom runner must have HTTPS settings active for proper secure
> communication. Ensure that any services your runner interacts with are properly configured for HTTPS. For example,
> make sure to set `THANOS_HTTPS=true` and similar environment variables for other services.

### Proxy Configuration

If you're having connection issues when running artemis-runner from the bundle in a corporate environment with a proxy, try adding your Artemis deployment to `no_proxy`:

```sh
# For SaaS deployment
export no_proxy=localhost,127.0.0.1,artemis.turintech.ai

# For internal deployment using domain name
export no_proxy=localhost,127.0.0.1,artemis.internal.domain.com

# For internal deployment using IP address
export no_proxy=localhost,127.0.0.1,192.168.41.1
```

The `no_proxy` environment variable (also works as `NO_PROXY`) ensures direct connections to Artemis, bypassing the
corporate proxy.

**Example with proxy environment**:

```sh
export no_proxy=localhost,127.0.0.1,artemis.turintech.ai

# Install and run
pip install --find-links wheels/ artemis-runner
artemis-runner --runner-name my-bundle-runner
```

## Sourcekon

### Installation

```sh
pip install --find-links wheels/ sourcekon
```

### Basic Usage

#### Profiling with Speedscope

To generate a sourcekon profile from a speedscope flame graph (created with `pyspy`), use the following command:

```sh
sourcekon speedscope \
--project_path /path/to/projectdir \
--executable_path /path/to/entrypoint.py \
--output_path /path/to/outputdir
```

#### Profiling with Intel™ VTune

To generate a sourcekon profile from VTune, use the following command:

```sh
sourcekon vtune \
--project_path=/path/to/project \
--executable_path /path/to/builtfile \
--output_path /path/to/outputdir/ \
--vtune_path /opt/intel/oneapi/vtune/2025.0/bin64/vtune \ # Change to match your own VTune location
--build_command "exit 0"
```

## Artemis Client

### Installation

```sh
pip install --find-links wheels/ artemis-client
```

### Configuration

Create a `.env` file in your working directory with the following variables:

```dosini
#########  THANOS  ###########
# Required for authentication
THANOS_USERNAME="<your artemis email address>"
THANOS_PASSWORD="<your artemis password>"
THANOS_CLIENT_ID=gfwRjq
THANOS_CLIENT_SECRET=xwJMpspatz
THANOS_GRANT_TYPE=password
THANOS_HOST=artemis.turintech.ai
THANOS_PORT=443
THANOS_POSTFIX=/turintech-thanos/api
THOR_HTTPS=True
THANOS_HTTPS=True

#########  FALCON  ###########
FALCON_HTTPS=True
FALCON_HOST=artemis.turintech.ai
FALCON_PORT=443
FALCON_POSTFIX=/turintech-falcon/api

#########  VISION  ###########
VISION_HTTPS=True
VISION_HOST=artemis.turintech.ai
VISION_PORT=443
VISION_POSTFIX=/turintech-vision/api
```

### Basic Usage

Here is a simple script calling Falcon (API for creating/managing projects) and Vision (API for interacting with LLMs).

```python
from artemis_client.falcon.client import FalconClient, FalconSettings
from artemis_client.vision.client import VisionClient, VisionSettings
from evoml_services.clients.thanos.client import ThanosSettings
from vision_models import LLMInferenceRequest
from vision_models.service.llm import LLMType
from vision_models.service.message import LLMConversationMessage, LLMRole


def falcon_client() -> FalconClient:
    falcon_settings = FalconSettings.with_env_prefix("falcon", _env_file=".env")
    thanos_settings = ThanosSettings.with_env_prefix("thanos", _env_file=".env")
    return FalconClient(falcon_settings, thanos_settings)


def vision_client() -> VisionClient:
    vision_settings = VisionSettings.with_env_prefix("vision", _env_file=".env")
    thanos_settings = ThanosSettings.with_env_prefix("thanos", _env_file=".env")
    return VisionClient(vision_settings, thanos_settings)


def main():
    f_client = falcon_client()

    print("Testing Falcon Client")
    project = f_client.get_projects("<use-an-id-of-an-existing-project>")
    print(f"Latest Project:\n {project.docs[0].model_dump_json(indent=2)}")

    print("===")

    v_client = vision_client()

    request = LLMInferenceRequest(
        model_type=LLMType.OPENAI_GPT_4_O,
        messages=[
            LLMConversationMessage(role=LLMRole.SYSTEM, content="You are a helpful assistant."),
            LLMConversationMessage(role=LLMRole.USER, content="What is the capital of France?"),
        ],
    )

    print("Testing Vision Client")
    response = v_client.ask(request)
    print(f"Response: {response.completions[0]}")


if __name__ == "__main__":
    main()
```
