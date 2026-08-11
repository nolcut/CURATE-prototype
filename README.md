# CURATE
CURATE (Composition, User-in-the-loop, Reuse, and Automated Task Execution) is a human-in-the-loop multi-agent system that manages scientific workflows
across their whole lifecycle, from a natural-language sketch through code generation,
deployment, and reuse. You describe a workflow in plain English; a set of LLM agents
drafts the task graph, decides which steps can be reused from a catalog, writes the
Python for the rest, deploys the result to a workflow management system, and monitors
the run. You review and approve at every stage.


## Configuration

All configuration lives in a `.env` file at the repo root. The easiest way to create it
is the setup script:

```
python3 setup_env.py
```

It walks through every variable in `.env.template` in order, grouped into three
sections, and prints a hint for each one telling you where to get the value. Press Enter
to accept the shown default. If a `.env` already exists it is read first and its values
become the defaults, so you can rerun the script to change one or two things without
retyping the rest. When it finishes it writes `.env` and warns you about any required
value still left empty.

The script only needs Python, not the project dependencies, so you can run it before
`uv sync` or before pulling the Docker image.

What you will be asked for:

- **LLM credentials.** `BEDROCK_API_KEY` and `AWS_REGION` for AWS Bedrock, the default
  provider. `ANTHROPIC_API_KEY` is optional and only used when you pass
  `--anthropic-api`. Bedrock also works with ordinary IAM credentials or an AWS profile
  in the environment instead of a Bedrock API key.
- **GitHub.** `GH_PAT`, a personal access token with repo and workflow scopes,
  `FAASR_GH_USERNAME`, and `FAASR_ACTION_REPO`, an existing repo where function code and
  Actions workflows are pushed.
- **S3 storage.** `FAASR_S3_ENDPOINT`, `FAASR_S3_BUCKET`, `FAASR_S3_REGION`,
  `S3_AccessKey`, and `S3_SecretKey`. This is where all data moving between workflow
  steps lives. The endpoint defaults to AWS S3 but any S3-compatible service works, such
  as MinIO or R2.

To fill the file in by hand instead, copy `.env.template` to `.env` and edit it. `.env`
is gitignored; do not commit it.

## Running

With Docker:

```
docker pull nolcut/curate
docker run -it -v "$PWD:/work" nolcut/curate faasr-agents
```

The container starts in a shell rather than launching the system, since a fresh container has
no credentials. Run `curate-setup` inside it to fill in `/app/.env` or pass a filled-out .env file with --env-file 

From source, with Python 3.13 and [uv](https://docs.astral.sh/uv/):
```
uv sync
uv run faasr-agents
```


### Flags

- `--sonnet` model tier, the default
- `--opus` larger model tier
- `--anthropic-api` use the Anthropic API instead of Bedrock
- `--debug` trace agent tool calls and keep the generated context directory on disk

## Output

Every run writes its workflow JSON and function files to `faasr_output/<workflow>/`,
whether it succeeded or not, so a failed run is still inspectable. Artifacts you pull
down during output review go to `downloads/<workflow>/`. At Gate 5, `export <dir>` writes
an evaluation bundle for the run: the prompt, the workflow, per-agent token costs, every
revision, and the full console log.

## Layout

```
src/faasr_agents/
  orchestrator.py   LangGraph state machine and gate routing
  cli.py            CLI entry point and gate rendering
  agents/           WCA, MCA (fca), MGA (fga), WDA
  prompts/          agent system prompts
  catalog/          BM25 catalog store and saved workflows
  faasr/            workflow JSON emit, validation, MGA context directory, FaaSr stubs
  deploy/           GitHub Actions registration and invocation
  tui/              code review, deploy tracker, output review viewers
  pricing.py        per-agent token and cost accounting
```

## Paper

```
@misc{cutler2026curateleveragingllmagents,
      title={CURATE: Leveraging LLM Agents to Compose, Catalog, and Deploy Reproducible Workflows}, 
      author={Nolan Cutler and Chia-Chen Kuo and Nanda Velugoti and Kathryn Newhart and Renato Figueiredo},
      year={2026},
      eprint={2608.04270},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2608.04270}, 
}
```
