"""Configuration file for the project."""

from typing import Any

import yaml
from pydantic import BaseModel

from games_sales import PROJECT_DIR


class ProjectConfig(BaseModel):
    """Represent project configuration parameters loaded from YAML.

    Handles feature specifications, catalog details, and experiment parameters.
    Supports environment-specific configuration overrides.
    """

    catalog_name: str
    schema_name: str
    experiment_name_basic: str
    experiment_name_fe: str
    data_source: dict[str, Any]
    preprocessing: dict[str, Any]
    target_column: str
    num_features: list[str]
    cat_features: list[str]
    features_from_lookup: list[str]
    model_parameters: dict[str, Any]

    @classmethod
    def from_yaml(cls, config_path: str, env: str = "dev") -> "ProjectConfig":
        """Load and parse configuration settings from a YAML file.

        :param config_path: Path to the YAML configuration file
        :param env: Environment name to load environment-specific settings
        :return: ProjectConfig instance initialized with parsed configuration
        """
        if env not in ["prd", "acc", "dev"]:
            raise ValueError(f"Invalid environment: {env}. Expected 'prd', 'acc', or 'dev'")

        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
            config_dict["catalog_name"] = config_dict[env]["catalog_name"]
            config_dict["schema_name"] = config_dict[env]["schema_name"]
            config_dict["data_source"]["local_path"] = (
                PROJECT_DIR / "data" / config_dict["data_source"]["file_name"]
            ).resolve()

            return cls(**config_dict)


class Tags(BaseModel):
    """Represents a set of tags for a Git commit.

    Contains information about the Git SHA, branch, and job run ID.
    """

    git_sha: str
    branch: str

    @classmethod
    def from_git_repo(cls, repo_path: str = PROJECT_DIR) -> "Tags":
        """Create Tags instance from a Git repository.

        :param repo_path: Path to the Git repository
        :return: Tags instance with git_sha and branch populated
        """
        from git import Repo

        try:
            repo = Repo(repo_path)
            git_sha = repo.head.commit.hexsha
            branch = repo.active_branch.name
        except Exception:
            git_sha = "unknown"
            branch = "unknown"

        return cls(git_sha=git_sha, branch=branch)
