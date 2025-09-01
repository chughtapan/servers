import logging
from pathlib import Path
from typing import Sequence, Optional, List, Dict
from mcp.server import Server
from mcp.server.session import ServerSession
from mcp.server.stdio import stdio_server
from mcp.server.elicitation import (
    elicit_with_validation,
    AcceptedElicitation,
    DeclinedElicitation,
)
from mcp.types import (
    ClientCapabilities,
    TextContent,
    Tool,
    ListRootsResult,
    RootsCapability,
)
from enum import Enum
import git
from pydantic import BaseModel, Field, create_model

# Default number of context lines to show in diff output
DEFAULT_CONTEXT_LINES = 3


class GitStatus(BaseModel):
    repo_path: str


class GitDiffUnstaged(BaseModel):
    repo_path: str
    context_lines: int = DEFAULT_CONTEXT_LINES


class GitDiffStaged(BaseModel):
    repo_path: str
    context_lines: int = DEFAULT_CONTEXT_LINES


class GitDiff(BaseModel):
    repo_path: str
    target: str
    context_lines: int = DEFAULT_CONTEXT_LINES


class GitCommit(BaseModel):
    repo_path: str
    message: str


class GitAdd(BaseModel):
    repo_path: str
    files: list[str]


class GitReset(BaseModel):
    repo_path: str


class GitLog(BaseModel):
    repo_path: str
    max_count: int = 10
    start_timestamp: Optional[str] = Field(
        None,
        description="Start timestamp for filtering commits. Accepts: ISO 8601 format (e.g., '2024-01-15T14:30:25'), relative dates (e.g., '2 weeks ago', 'yesterday'), or absolute dates (e.g., '2024-01-15', 'Jan 15 2024')",
    )
    end_timestamp: Optional[str] = Field(
        None,
        description="End timestamp for filtering commits. Accepts: ISO 8601 format (e.g., '2024-01-15T14:30:25'), relative dates (e.g., '2 weeks ago', 'yesterday'), or absolute dates (e.g., '2024-01-15', 'Jan 15 2024')",
    )


class GitCreateBranch(BaseModel):
    repo_path: str
    branch_name: str
    base_branch: str | None = None


class GitCheckout(BaseModel):
    repo_path: str
    branch_name: str


class GitShow(BaseModel):
    repo_path: str
    revision: str


class GitInit(BaseModel):
    repo_path: str


class GitBranch(BaseModel):
    repo_path: str = Field(
        ...,
        description="The path to the Git repository.",
    )
    branch_type: str = Field(
        ...,
        description="Whether to list local branches ('local'), remote branches ('remote') or all branches('all').",
    )
    contains: Optional[str] = Field(
        None,
        description="The commit sha that branch should contain. Do not pass anything to this param if no commit sha is specified",
    )
    not_contains: Optional[str] = Field(
        None,
        description="The commit sha that branch should NOT contain. Do not pass anything to this param if no commit sha is specified",
    )


class GitTools(str, Enum):
    STATUS = "git_status"
    DIFF_UNSTAGED = "git_diff_unstaged"
    DIFF_STAGED = "git_diff_staged"
    DIFF = "git_diff"
    COMMIT = "git_commit"
    ADD = "git_add"
    RESET = "git_reset"
    LOG = "git_log"
    CREATE_BRANCH = "git_create_branch"
    CHECKOUT = "git_checkout"
    SHOW = "git_show"
    INIT = "git_init"
    BRANCH = "git_branch"


def git_status(repo: git.Repo) -> str:
    return repo.git.status()


def git_diff_unstaged(
    repo: git.Repo, context_lines: int = DEFAULT_CONTEXT_LINES
) -> str:
    return repo.git.diff(f"--unified={context_lines}")


def git_diff_staged(repo: git.Repo, context_lines: int = DEFAULT_CONTEXT_LINES) -> str:
    return repo.git.diff(f"--unified={context_lines}", "--cached")


def git_diff(
    repo: git.Repo, target: str, context_lines: int = DEFAULT_CONTEXT_LINES
) -> str:
    return repo.git.diff(f"--unified={context_lines}", target)


def git_commit(repo: git.Repo, message: str) -> str:
    commit = repo.index.commit(message)
    return f"Changes committed successfully with hash {commit.hexsha}"


def git_add(repo: git.Repo, files: list[str]) -> str:
    if files == ["."]:
        repo.git.add(".")
    else:
        repo.index.add(files)
    return "Files staged successfully"


def git_reset(repo: git.Repo) -> str:
    repo.index.reset()
    return "All staged changes reset"


def git_log(
    repo: git.Repo,
    max_count: int = 10,
    start_timestamp: Optional[str] = None,
    end_timestamp: Optional[str] = None,
) -> list[str]:
    if start_timestamp or end_timestamp:
        # Use git log command with date filtering
        args = []
        if start_timestamp:
            args.extend(["--since", start_timestamp])
        if end_timestamp:
            args.extend(["--until", end_timestamp])
        args.extend(["--format=%H%n%an%n%ad%n%s%n"])

        log_output = repo.git.log(*args).split("\n")

        log = []
        # Process commits in groups of 4 (hash, author, date, message)
        for i in range(0, len(log_output), 4):
            if i + 3 < len(log_output) and len(log) < max_count:
                log.append(
                    f"Commit: {log_output[i]}\n"
                    f"Author: {log_output[i+1]}\n"
                    f"Date: {log_output[i+2]}\n"
                    f"Message: {log_output[i+3]}\n"
                )
        return log
    else:
        # Use existing logic for simple log without date filtering
        commits = list(repo.iter_commits(max_count=max_count))
        log = []
        for commit in commits:
            log.append(
                f"Commit: {commit.hexsha!r}\n"
                f"Author: {commit.author!r}\n"
                f"Date: {commit.authored_datetime}\n"
                f"Message: {commit.message!r}\n"
            )
        return log


def git_create_branch(
    repo: git.Repo, branch_name: str, base_branch: str | None = None
) -> str:
    if base_branch:
        base = repo.references[base_branch]
    else:
        base = repo.active_branch

    repo.create_head(branch_name, base)
    return f"Created branch '{branch_name}' from '{base.name}'"


def git_checkout(repo: git.Repo, branch_name: str) -> str:
    repo.git.checkout(branch_name)
    return f"Switched to branch '{branch_name}'"


def git_init(repo_path: str) -> str:
    try:
        repo = git.Repo.init(path=repo_path, mkdir=True)
        return f"Initialized empty Git repository in {repo.git_dir}"
    except Exception as e:
        return f"Error initializing repository: {str(e)}"


def git_show(repo: git.Repo, revision: str) -> str:
    commit = repo.commit(revision)
    output = [
        f"Commit: {commit.hexsha!r}\n"
        f"Author: {commit.author!r}\n"
        f"Date: {commit.authored_datetime!r}\n"
        f"Message: {commit.message!r}\n"
    ]
    if commit.parents:
        parent = commit.parents[0]
        diff = parent.diff(commit, create_patch=True)
    else:
        diff = commit.diff(git.NULL_TREE, create_patch=True)
    for d in diff:
        output.append(f"\n--- {d.a_path}\n+++ {d.b_path}\n")
        output.append(d.diff.decode("utf-8"))
    return "".join(output)


def git_branch(
    repo: git.Repo,
    branch_type: str,
    contains: str | None = None,
    not_contains: str | None = None,
) -> str:
    match contains:
        case None:
            contains_sha = (None,)
        case _:
            contains_sha = ("--contains", contains)

    match not_contains:
        case None:
            not_contains_sha = (None,)
        case _:
            not_contains_sha = ("--no-contains", not_contains)

    match branch_type:
        case "local":
            b_type = None
        case "remote":
            b_type = "-r"
        case "all":
            b_type = "-a"
        case _:
            return f"Invalid branch type: {branch_type}"

    # None value will be auto deleted by GitPython
    branch_info = repo.git.branch(b_type, *contains_sha, *not_contains_sha)

    return branch_info


def parse_status_output(status_output: str) -> Dict[str, List[str]]:
    """Parse git status --porcelain output into categorized files"""
    tracked_files = []
    untracked_files = []

    for line in status_output.split("\n"):
        if line:
            status = line[:2]
            filepath = line[3:]

            # Check for untracked files
            if status == "??":
                untracked_files.append(filepath)
            # Check for modified (M) or deleted (D) files in working tree
            elif status[1] in ["M", "D"]:
                tracked_files.append(filepath)

    return {"tracked": tracked_files, "untracked": untracked_files}


def get_file_change_type(repo: git.Repo, filepath: str) -> str:
    """Determine if a staged file is new, modified, or deleted"""
    # Get staged changes from HEAD's perspective
    head = repo.head.commit
    staged_items = head.diff(None, staged=True)

    for item in staged_items:
        if item.a_path == filepath:
            if item.change_type == "A":
                return "new file"
            elif item.change_type == "D":
                return "deleted"
            elif item.change_type == "M":
                return "modified"
            elif item.change_type == "R":
                return "renamed"

    # If not in staged, check if it's untracked (will be new when staged)
    status = repo.git.status("--porcelain", filepath)
    if status.startswith("??"):
        return "new file"

    return "modified"  # Default


def format_staged_files_display(repo: git.Repo, staged_files: list) -> str:
    """Format staged files in git-style display"""
    if not staged_files:
        return "No files staged for commit"

    # Get current branch
    try:
        branch_name = repo.active_branch.name
    except TypeError:
        # Detached HEAD state
        branch_name = "HEAD (detached)"

    lines = [f"On branch {branch_name}", "Changes to be committed:"]

    for filepath in staged_files:
        change_type = get_file_change_type(repo, filepath)
        lines.append(f"  {change_type}:   {filepath}")

    return "\n".join(lines)


def supports_elicitation(server: Server) -> bool:
    """Check if current client supports elicitation"""
    session = server.request_context.session
    return isinstance(session, ServerSession) and session.check_client_capability(
        ClientCapabilities(elicitation={})
    )


async def git_commit_with_elicitation(
    repo: git.Repo, message: str, session: ServerSession, request_id: str | None = None
) -> str:
    """Git commit with interactive elicitation showing staged files"""
    # Get list of staged files
    head = repo.head.commit
    staged_items = head.diff(None, staged=True)
    staged_files = [item.a_path for item in staged_items]

    # If no files are staged, return early
    if not staged_files:
        return "No files staged for commit. Use git add to stage files first."

    # Create git-style formatted display of staged files
    staged_files_display = format_staged_files_display(repo, staged_files)

    # Capture the message value for use in the schema
    default_message = message

    # Create schema with only the commit message as editable
    CommitElicitationSchema = create_model(
        "CommitElicitationSchema",
        message=(str, Field(default=default_message, description="Commit message")),
    )

    # Perform elicitation with git-style display
    result = await elicit_with_validation(
        session=session,
        message=staged_files_display,
        schema=CommitElicitationSchema,
        related_request_id=request_id,
    )

    # Handle elicitation result
    if isinstance(result, AcceptedElicitation):
        # Proceed with commit
        commit = repo.index.commit(result.data.message)
        return f"Changes committed successfully with hash {commit.hexsha}"
    elif isinstance(result, DeclinedElicitation):
        return "Commit declined by user"
    else:  # CancelledElicitation
        return "Commit cancelled by user"


async def git_add_with_elicitation(
    repo: git.Repo,
    files: list[str],
    session: ServerSession,
    request_id: str | None = None,
) -> str:
    """Git add with interactive file selection showing separate groups for tracked and untracked files"""
    # If "." was provided, fall back to non-elicitation behavior
    if files == ["."]:
        result = git_add(repo, files)
        return result

    # Get unstaged files categorized
    status_output = repo.git.status("--porcelain")
    file_categories = parse_status_output(status_output)

    tracked_files = file_categories["tracked"]
    untracked_files = file_categories["untracked"]

    # If no files to stage, return early
    if not tracked_files and not untracked_files:
        return "No changes to stage"

    # Determine which files should be selected by default
    # Only select requested files that are actually unstaged
    default_tracked = []
    default_untracked = []

    # Only include requested files that are actually unstaged
    default_tracked = [f for f in files if f in tracked_files]
    default_untracked = [f for f in files if f in untracked_files]

    # Create dynamic schema for file selection with separate groups
    model_fields = {}

    # Add tracked files field if there are any
    if tracked_files:
        model_fields["tracked_files_to_stage"] = (
            List[str],
            Field(
                default=default_tracked,
                description="Modified/deleted files",
                json_schema_extra={"items": {"type": "string", "enum": tracked_files}},
            ),
        )

    # Add untracked files field if there are any
    if untracked_files:
        model_fields["untracked_files_to_stage"] = (
            List[str],
            Field(
                default=default_untracked,
                description="New files",
                json_schema_extra={
                    "items": {"type": "string", "enum": untracked_files}
                },
            ),
        )

    AddElicitationSchema = create_model("AddElicitationSchema", **model_fields)

    # Create elicitation message
    messages = []
    messages.append("Select files to stage:")
    elicitation_message = "\n".join(messages)

    # Perform elicitation
    result = await elicit_with_validation(
        session=session,
        message=elicitation_message,
        schema=AddElicitationSchema,
        related_request_id=request_id,
    )

    # Handle elicitation result
    if isinstance(result, AcceptedElicitation):
        # Collect all selected files from both groups
        selected_files = []

        if tracked_files and hasattr(result.data, "tracked_files_to_stage"):
            selected_files.extend(result.data.tracked_files_to_stage)

        if untracked_files and hasattr(result.data, "untracked_files_to_stage"):
            selected_files.extend(result.data.untracked_files_to_stage)

        if selected_files:
            # Stage selected files
            repo.index.add(selected_files)

            # Create detailed return message
            if len(selected_files) <= 5:
                file_list = ", ".join(selected_files)
                return f"Staged {len(selected_files)} file(s): {file_list}"
            else:
                # For many files, show first few and count
                file_list = ", ".join(selected_files[:3])
                return f"Staged {len(selected_files)} file(s): {file_list}, ... and {len(selected_files) - 3} more"
        else:
            return "No files selected for staging"
    elif isinstance(result, DeclinedElicitation):
        return "File staging declined by user"
    else:  # CancelledElicitation
        return "File staging cancelled by user"


async def serve(repository: Path | None) -> None:
    logger = logging.getLogger(__name__)

    if repository is not None:
        try:
            git.Repo(repository)
            logger.info(f"Using repository at {repository}")
        except git.InvalidGitRepositoryError:
            logger.error(f"{repository} is not a valid Git repository")
            return

    server = Server("mcp-git")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=GitTools.STATUS,
                description="Shows the working tree status",
                inputSchema=GitStatus.model_json_schema(),
            ),
            Tool(
                name=GitTools.DIFF_UNSTAGED,
                description="Shows changes in the working directory that are not yet staged",
                inputSchema=GitDiffUnstaged.model_json_schema(),
            ),
            Tool(
                name=GitTools.DIFF_STAGED,
                description="Shows changes that are staged for commit",
                inputSchema=GitDiffStaged.model_json_schema(),
            ),
            Tool(
                name=GitTools.DIFF,
                description="Shows differences between branches or commits",
                inputSchema=GitDiff.model_json_schema(),
            ),
            Tool(
                name=GitTools.COMMIT,
                description="Records changes to the repository",
                inputSchema=GitCommit.model_json_schema(),
            ),
            Tool(
                name=GitTools.ADD,
                description="Adds file contents to the staging area",
                inputSchema=GitAdd.model_json_schema(),
            ),
            Tool(
                name=GitTools.RESET,
                description="Unstages all staged changes",
                inputSchema=GitReset.model_json_schema(),
            ),
            Tool(
                name=GitTools.LOG,
                description="Shows the commit logs",
                inputSchema=GitLog.model_json_schema(),
            ),
            Tool(
                name=GitTools.CREATE_BRANCH,
                description="Creates a new branch from an optional base branch",
                inputSchema=GitCreateBranch.model_json_schema(),
            ),
            Tool(
                name=GitTools.CHECKOUT,
                description="Switches branches",
                inputSchema=GitCheckout.model_json_schema(),
            ),
            Tool(
                name=GitTools.SHOW,
                description="Shows the contents of a commit",
                inputSchema=GitShow.model_json_schema(),
            ),
            Tool(
                name=GitTools.INIT,
                description="Initialize a new Git repository",
                inputSchema=GitInit.model_json_schema(),
            ),
            Tool(
                name=GitTools.BRANCH,
                description="List Git branches",
                inputSchema=GitBranch.model_json_schema(),
            ),
        ]

    async def list_repos() -> Sequence[str]:
        async def by_roots() -> Sequence[str]:
            if not isinstance(server.request_context.session, ServerSession):
                raise TypeError(
                    "server.request_context.session must be a ServerSession"
                )

            if not server.request_context.session.check_client_capability(
                ClientCapabilities(roots=RootsCapability())
            ):
                return []

            roots_result: ListRootsResult = (
                await server.request_context.session.list_roots()
            )
            logger.debug(f"Roots result: {roots_result}")
            repo_paths = []
            for root in roots_result.roots:
                path = root.uri.path
                try:
                    git.Repo(path)
                    repo_paths.append(str(path))
                except git.InvalidGitRepositoryError:
                    pass
            return repo_paths

        def by_commandline() -> Sequence[str]:
            return [str(repository)] if repository is not None else []

        cmd_repos = by_commandline()
        root_repos = await by_roots()
        return [*root_repos, *cmd_repos]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        repo_path = Path(arguments["repo_path"])

        # Handle git init separately since it doesn't require an existing repo
        if name == GitTools.INIT:
            result = git_init(str(repo_path))
            return [TextContent(type="text", text=result)]

        # For all other commands, we need an existing repo
        repo = git.Repo(repo_path)

        match name:
            case GitTools.STATUS:
                status = git_status(repo)
                return [TextContent(type="text", text=f"Repository status:\n{status}")]

            case GitTools.DIFF_UNSTAGED:
                diff = git_diff_unstaged(
                    repo, arguments.get("context_lines", DEFAULT_CONTEXT_LINES)
                )
                return [TextContent(type="text", text=f"Unstaged changes:\n{diff}")]

            case GitTools.DIFF_STAGED:
                diff = git_diff_staged(
                    repo, arguments.get("context_lines", DEFAULT_CONTEXT_LINES)
                )
                return [TextContent(type="text", text=f"Staged changes:\n{diff}")]

            case GitTools.DIFF:
                diff = git_diff(
                    repo,
                    arguments["target"],
                    arguments.get("context_lines", DEFAULT_CONTEXT_LINES),
                )
                return [
                    TextContent(
                        type="text", text=f"Diff with {arguments['target']}:\n{diff}"
                    )
                ]

            case GitTools.COMMIT:
                if supports_elicitation(server):
                    session = server.request_context.session
                    request_id = getattr(server.request_context, "request_id", None)
                    result = await git_commit_with_elicitation(
                        repo, arguments["message"], session, request_id
                    )
                else:
                    result = git_commit(repo, arguments["message"])
                return [TextContent(type="text", text=result)]

            case GitTools.ADD:
                if supports_elicitation(server):
                    session = server.request_context.session
                    request_id = getattr(server.request_context, "request_id", None)
                    result = await git_add_with_elicitation(
                        repo, arguments["files"], session, request_id
                    )
                else:
                    result = git_add(repo, arguments["files"])
                return [TextContent(type="text", text=result)]

            case GitTools.RESET:
                result = git_reset(repo)
                return [TextContent(type="text", text=result)]

            # Update the LOG case:
            case GitTools.LOG:
                log = git_log(
                    repo,
                    arguments.get("max_count", 10),
                    arguments.get("start_timestamp"),
                    arguments.get("end_timestamp"),
                )
                return [
                    TextContent(type="text", text="Commit history:\n" + "\n".join(log))
                ]

            case GitTools.CREATE_BRANCH:
                result = git_create_branch(
                    repo, arguments["branch_name"], arguments.get("base_branch")
                )
                return [TextContent(type="text", text=result)]

            case GitTools.CHECKOUT:
                result = git_checkout(repo, arguments["branch_name"])
                return [TextContent(type="text", text=result)]

            case GitTools.SHOW:
                result = git_show(repo, arguments["revision"])
                return [TextContent(type="text", text=result)]

            case GitTools.BRANCH:
                result = git_branch(
                    repo,
                    arguments.get("branch_type", "local"),
                    arguments.get("contains", None),
                    arguments.get("not_contains", None),
                )
                return [TextContent(type="text", text=result)]

            case _:
                raise ValueError(f"Unknown tool: {name}")

    options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, options, raise_exceptions=True)
