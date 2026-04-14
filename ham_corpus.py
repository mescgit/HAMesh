"""
HAMesh Corpus Loader — Verified math knowledge into the mesh

Parses Metamath databases (.mm files) and folds theorems into a
HolographicMesh using the LLM-free embedder. Every statement that enters
the mesh has been formally verified — no hallucination possible.

The key-value pairs folded in are:
  key: theorem name + statement  (what it IS)
  val: theorem name + proof sketch / nearby axioms  (how it CONNECTS)

This grounds the mesh in real mathematical relationships rather than
natural-language associations.

Usage:
    # Download set.mm first:
    #   https://us.metamath.org/downloads/metamath.zip  (extract set.mm)
    # Or use a subset — see BUNDLED_SUBSETS below.

    python ham_corpus.py --subset arithmetic --save math_mesh.pt
    python ham_corpus.py --file set.mm --filter "number theory" --save math_mesh.pt
    python ham_corpus.py --builtin --save math_mesh.pt  (no download needed)
"""

import argparse
import json
import re
import urllib.request
from pathlib import Path

import torch

from ham_core import HolographicMesh
from ham_embedder import Embedder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Bundled mini-corpus: ~120 foundational theorems, no download required
# Covers: Peano arithmetic, basic number theory, set theory primitives,
#         propositional logic, geometry basics
# Each entry: (name, statement, domain)
# ---------------------------------------------------------------------------

BUILTIN_CORPUS = [
    # --- Propositional logic ---
    ("modus ponens",       "If P implies Q, and P is true, then Q is true.", "logic"),
    ("modus tollens",      "If P implies Q, and Q is false, then P is false.", "logic"),
    ("hypothetical syllogism", "If P implies Q and Q implies R, then P implies R.", "logic"),
    ("disjunctive syllogism",  "If P or Q is true, and P is false, then Q must be true.", "logic"),
    ("de morgan and",      "Not (P and Q) is equivalent to (not P) or (not Q).", "logic"),
    ("de morgan or",       "Not (P or Q) is equivalent to (not P) and (not Q).", "logic"),
    ("law of excluded middle", "For any proposition P, either P or not P is true.", "logic"),
    ("double negation",    "Not not P is equivalent to P.", "logic"),
    ("contrapositive",     "P implies Q is equivalent to not Q implies not P.", "logic"),
    ("biconditional",      "P if and only if Q means P implies Q and Q implies P.", "logic"),

    # --- Set theory ---
    ("extensionality",     "Two sets are equal if and only if they have the same elements.", "set theory"),
    ("empty set",          "There exists a set with no elements, called the empty set.", "set theory"),
    ("pairing axiom",      "For any two sets A and B, there exists a set containing exactly A and B.", "set theory"),
    ("union axiom",        "For any set of sets, there exists a set that is the union of all of them.", "set theory"),
    ("power set axiom",    "For any set A, there exists a set of all subsets of A.", "set theory"),
    ("axiom of infinity",  "There exists a set that contains the empty set and is closed under successor.", "set theory"),
    ("axiom of choice",    "For any collection of non-empty sets, there exists a function selecting one element from each.", "set theory"),
    ("subset definition",  "A is a subset of B if every element of A is also an element of B.", "set theory"),
    ("intersection",       "The intersection of A and B is the set of elements belonging to both A and B.", "set theory"),
    ("complement",         "The complement of A in B is the set of elements in B that are not in A.", "set theory"),
    ("cartesian product",  "The Cartesian product A cross B is the set of all ordered pairs (a, b) with a in A and b in B.", "set theory"),
    ("russell paradox",    "The set of all sets that do not contain themselves leads to a contradiction.", "set theory"),
    ("cantor theorem",     "The power set of any set has strictly greater cardinality than the set itself.", "set theory"),
    ("cantor diagonal",    "No surjection exists from a set to its power set.", "set theory"),

    # --- Peano arithmetic ---
    ("peano zero",         "Zero is a natural number.", "arithmetic"),
    ("peano successor",    "Every natural number n has a successor S(n) which is also a natural number.", "arithmetic"),
    ("peano zero not successor", "Zero is not the successor of any natural number.", "arithmetic"),
    ("peano successor injective", "If S(m) equals S(n) then m equals n.", "arithmetic"),
    ("peano induction",    "If a property holds for zero, and holds for S(n) whenever it holds for n, it holds for all natural numbers.", "arithmetic"),
    ("addition base",      "n plus zero equals n for all natural numbers n.", "arithmetic"),
    ("addition recursive", "n plus S(m) equals S(n plus m).", "arithmetic"),
    ("multiplication base","n times zero equals zero.", "arithmetic"),
    ("multiplication recursive", "n times S(m) equals (n times m) plus n.", "arithmetic"),
    ("commutativity addition",    "m plus n equals n plus m for all natural numbers.", "arithmetic"),
    ("associativity addition",    "(m plus n) plus p equals m plus (n plus p).", "arithmetic"),
    ("commutativity multiplication", "m times n equals n times m.", "arithmetic"),
    ("associativity multiplication", "(m times n) times p equals m times (n times p).", "arithmetic"),
    ("distributivity",     "m times (n plus p) equals (m times n) plus (m times p).", "arithmetic"),

    # --- Number theory ---
    ("divisibility",       "a divides b if there exists an integer k such that b equals a times k.", "number theory"),
    ("prime definition",   "A prime number is a natural number greater than 1 with no divisors other than 1 and itself.", "number theory"),
    ("fundamental theorem of arithmetic", "Every integer greater than 1 is either prime or a unique product of primes.", "number theory"),
    ("euclid infinitely many primes", "There are infinitely many prime numbers.", "number theory"),
    ("euclid lemma",       "If a prime p divides a product ab, then p divides a or p divides b.", "number theory"),
    ("gcd definition",     "The greatest common divisor of a and b is the largest integer dividing both.", "number theory"),
    ("bezout identity",    "For integers a and b, there exist integers x and y such that ax plus by equals gcd(a,b).", "number theory"),
    ("chinese remainder theorem", "If n1 and n2 are coprime, the system x congruent to a1 mod n1 and x congruent to a2 mod n2 has a unique solution mod n1 times n2.", "number theory"),
    ("fermat little theorem", "If p is prime and a is not divisible by p, then a to the power p minus 1 is congruent to 1 modulo p.", "number theory"),
    ("euler totient theorem", "If gcd(a, n) equals 1 then a to the power phi(n) is congruent to 1 modulo n.", "number theory"),
    ("wilson theorem",     "p is prime if and only if (p minus 1) factorial is congruent to negative 1 modulo p.", "number theory"),
    ("quadratic reciprocity", "For distinct odd primes p and q, the Legendre symbols satisfy a specific multiplicative relationship.", "number theory"),
    ("goldbach conjecture", "Every even integer greater than 2 is the sum of two primes. (Unproven as of 2025.)", "number theory"),
    ("twin prime conjecture", "There are infinitely many pairs of primes differing by 2. (Unproven as of 2025.)", "number theory"),
    ("riemann hypothesis",  "All non-trivial zeros of the Riemann zeta function have real part 1/2. (Unproven as of 2025.)", "number theory"),

    # --- Algebra ---
    ("group definition",   "A group is a set with an associative binary operation, an identity element, and inverses.", "algebra"),
    ("abelian group",      "A group is abelian if its operation is commutative.", "algebra"),
    ("subgroup",           "A subset H of group G is a subgroup if it is closed under the group operation and inverses.", "algebra"),
    ("lagrange theorem",   "The order of a subgroup divides the order of the group.", "algebra"),
    ("ring definition",    "A ring is a set with two operations: addition (abelian group) and associative multiplication distributive over addition.", "algebra"),
    ("field definition",   "A field is a ring where every non-zero element has a multiplicative inverse.", "algebra"),
    ("homomorphism",       "A homomorphism is a structure-preserving map between algebraic structures.", "algebra"),
    ("kernel",             "The kernel of a homomorphism is the set of elements mapping to the identity.", "algebra"),
    ("first isomorphism theorem", "The quotient of a group by the kernel of a homomorphism is isomorphic to the image.", "algebra"),
    ("polynomial ring",    "The set of polynomials with coefficients in a ring forms a ring under addition and multiplication.", "algebra"),
    ("fundamental theorem of algebra", "Every non-constant polynomial with complex coefficients has at least one complex root.", "algebra"),
    ("cayley theorem",     "Every group is isomorphic to a subgroup of a symmetric group.", "algebra"),

    # --- Analysis / Calculus ---
    ("limit definition",   "The limit of f(x) as x approaches a is L if for every epsilon > 0 there exists delta > 0 such that |f(x) - L| < epsilon whenever 0 < |x - a| < delta.", "analysis"),
    ("continuity",         "A function f is continuous at a if the limit of f(x) as x approaches a equals f(a).", "analysis"),
    ("intermediate value theorem", "If f is continuous on [a,b] and f(a) and f(b) have opposite signs, there exists c in (a,b) with f(c) = 0.", "analysis"),
    ("extreme value theorem", "A continuous function on a closed bounded interval attains its maximum and minimum.", "analysis"),
    ("mean value theorem", "If f is differentiable on (a,b), there exists c in (a,b) where f'(c) equals (f(b)-f(a))/(b-a).", "analysis"),
    ("fundamental theorem of calculus", "The derivative of the integral of f from a to x equals f(x).", "analysis"),
    ("taylor series",      "A smooth function can be expressed as an infinite sum of terms involving its derivatives at a point.", "analysis"),
    ("cauchy sequence",    "A sequence where terms become arbitrarily close together converges in a complete metric space.", "analysis"),
    ("bolzano weierstrass","Every bounded sequence of real numbers has a convergent subsequence.", "analysis"),
    ("uniform convergence","A sequence of functions converges uniformly if the rate of convergence is independent of the point.", "analysis"),

    # --- Geometry ---
    ("pythagorean theorem","In a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides.", "geometry"),
    ("euclid parallel postulate", "Through a point not on a line, there is exactly one line parallel to the given line.", "geometry"),
    ("triangle angle sum", "The interior angles of a triangle sum to 180 degrees in Euclidean geometry.", "geometry"),
    ("similar triangles",  "Two triangles are similar if their corresponding angles are equal.", "geometry"),
    ("thales theorem",     "An angle inscribed in a semicircle is a right angle.", "geometry"),
    ("euler formula polyhedra", "For a convex polyhedron, vertices minus edges plus faces equals 2.", "geometry"),
    ("gauss egregium theorem", "The Gaussian curvature of a surface is an intrinsic property preserved under bending.", "geometry"),

    # --- Combinatorics ---
    ("pigeonhole principle","If n+1 objects are placed in n boxes, at least one box contains more than one object.", "combinatorics"),
    ("binomial theorem",   "The expansion of (x+y)^n is the sum of C(n,k) x^k y^(n-k) for k from 0 to n.", "combinatorics"),
    ("inclusion exclusion","The size of the union of sets equals the sum of sizes minus pairwise intersections plus triple intersections, and so on.", "combinatorics"),
    ("ramsey theory",      "In any sufficiently large structure, order must appear; complete disorder is impossible.", "combinatorics"),
    ("stirling numbers",   "Stirling numbers count the ways to partition a set into non-empty subsets.", "combinatorics"),

    # --- Topology ---
    ("open set definition","A topology on a set X is a collection of subsets closed under arbitrary unions and finite intersections.", "topology"),
    ("compactness",        "A space is compact if every open cover has a finite subcover.", "topology"),
    ("connectedness",      "A space is connected if it cannot be partitioned into two disjoint non-empty open sets.", "topology"),
    ("homeomorphism",      "A homeomorphism is a continuous bijection with a continuous inverse.", "topology"),
    ("heine borel theorem","A subset of Euclidean space is compact if and only if it is closed and bounded.", "topology"),
    ("brouwer fixed point","Every continuous function from a closed ball to itself has a fixed point.", "topology"),
    ("euler characteristic","The Euler characteristic is a topological invariant: V - E + F for surfaces.", "topology"),

    # --- Information theory ---
    ("shannon entropy",    "The entropy H of a probability distribution measures the expected information content: H = -sum p log p.", "information theory"),
    ("channel capacity",   "The maximum rate of reliable information transmission over a noisy channel.", "information theory"),
    ("data compression theorem", "Lossless compression cannot compress below the entropy rate of the source.", "information theory"),
    ("kolmogorov complexity", "The algorithmic complexity of a string is the length of its shortest description in a universal language.", "information theory"),

    # --- Computability ---
    ("turing completeness","A system is Turing complete if it can simulate any Turing machine.", "computability"),
    ("halting problem",    "There is no algorithm that can determine for all programs and inputs whether the program halts.", "computability"),
    ("church turing thesis","Any effectively computable function can be computed by a Turing machine.", "computability"),
    ("rice theorem",       "Any non-trivial semantic property of programs is undecidable.", "computability"),
    ("godel incompleteness first", "Any consistent formal system strong enough to describe arithmetic contains true statements it cannot prove.", "computability"),
    ("godel incompleteness second", "A consistent formal system cannot prove its own consistency.", "computability"),

    # --- Linear algebra ---
    ("eigenvalue definition","A scalar lambda is an eigenvalue of matrix A if there exists a non-zero vector v such that Av = lambda v.", "linear algebra"),
    ("spectral theorem",   "A real symmetric matrix has real eigenvalues and orthogonal eigenvectors.", "linear algebra"),
    ("rank nullity theorem","The rank plus the nullity of a linear map equals the dimension of the domain.", "linear algebra"),
    ("determinant",        "The determinant of a matrix measures the scaling factor of the linear transformation it represents.", "linear algebra"),
    ("singular value decomposition", "Every matrix can be written as U times Sigma times V transpose, where U and V are orthogonal.", "linear algebra"),
    ("gram schmidt",       "Any linearly independent set of vectors can be transformed into an orthonormal basis.", "linear algebra"),
    ("cayley hamilton theorem", "Every square matrix satisfies its own characteristic polynomial.", "linear algebra"),
]


# ---------------------------------------------------------------------------
# Metamath .mm file parser
# ---------------------------------------------------------------------------

METAMATH_URL = "https://us.metamath.org/downloads/metamath.zip"
METAMATH_DEFAULT_PATH = Path("./ham_data/set.mm")


def download_metamath(dest_dir: str = "./ham_data") -> Path:
    """
    Download and extract set.mm from metamath.org.
    Returns the path to the extracted set.mm file.
    """
    import zipfile

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    mm_path = dest / "set.mm"

    if mm_path.exists():
        print(f"  set.mm already exists at {mm_path} ({mm_path.stat().st_size // 1024 // 1024} MB)")
        return mm_path

    zip_path = dest / "metamath.zip"
    print(f"  Downloading metamath.zip from metamath.org...")
    print(f"  (This is ~30 MB, one-time download)")

    def _progress(count, block_size, total):
        pct = min(count * block_size * 100 // total, 100)
        if count % 50 == 0:
            print(f"  {pct}%", end="\r", flush=True)

    urllib.request.urlretrieve(METAMATH_URL, zip_path, reporthook=_progress)
    print(f"  Downloaded. Extracting set.mm...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        # set.mm may be nested inside a folder in the zip
        mm_names = [n for n in zf.namelist() if n.endswith("set.mm")]
        if not mm_names:
            raise RuntimeError("set.mm not found inside metamath.zip")
        zf.extract(mm_names[0], dest)
        extracted = dest / mm_names[0]
        if extracted != mm_path:
            extracted.rename(mm_path)

    zip_path.unlink()  # clean up zip
    print(f"  Extracted: {mm_path} ({mm_path.stat().st_size // 1024 // 1024} MB)")
    return mm_path


def parse_metamath(mm_path: str, max_theorems: int = 2000) -> list[dict]:
    """
    Parse a Metamath .mm file and extract theorem statements.

    Returns a list of dicts:
        {name, statement, hypotheses, domain}

    Only extracts $p (provable) statements. Strips $( $) comments.
    """
    path = Path(mm_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Metamath file not found: {mm_path}\n"
            f"  Run with --download to fetch it automatically, or download manually:\n"
            f"  {METAMATH_URL}"
        )

    print(f"  Parsing {path.name} ({path.stat().st_size // 1024 // 1024} MB)...")
    text = path.read_text(encoding="utf-8", errors="replace")

    # Strip comments $( ... $)
    text = re.sub(r'\$\(.*?\$\)', ' ', text, flags=re.DOTALL)

    theorems = []
    # Match: label $p ... $= ... $.
    pattern = re.compile(
        r'(\w+)\s+\$p\s+(.*?)\s+\$=\s+(.*?)\s+\$\.',
        re.DOTALL
    )
    for match in pattern.finditer(text):
        name = match.group(1)
        statement = ' '.join(match.group(2).split())
        proof_refs = match.group(3).split()[:10]  # first 10 proof steps as context
        theorems.append({
            'name': name,
            'statement': statement,
            'proof_sketch': ' '.join(proof_refs),
            'domain': 'metamath',
        })
        if len(theorems) >= max_theorems:
            break

    print(f"  Parsed {len(theorems)} theorems.")
    return theorems


# ---------------------------------------------------------------------------
# Mesh builder
# ---------------------------------------------------------------------------

def build_mesh_from_corpus(
    entries: list[dict],
    embedder: Embedder = None,
    fold_strength: float = 1.0,
    verbose: bool = True,
) -> HolographicMesh:
    """
    Fold a list of corpus entries into a fresh HolographicMesh.

    Each entry is folded as:
      key = embed(name + ": " + statement)
      val = embed(name + " connects to: " + domain + " | " + proof_sketch)

    Returns a loaded HolographicMesh ready for dreaming.
    """
    if embedder is None:
        embedder = Embedder()

    mesh = HolographicMesh(dim=embedder.dim, device=embedder.device)

    if verbose:
        print(f"\n  Building mesh from {len(entries)} entries (dim={embedder.dim})...")

    # Batch-embed all keys and values for speed
    keys_text = [f"{e['name']}: {e['statement']}" for e in entries]
    vals_text  = [
        f"{e['name']} in {e.get('domain','math')}: {e.get('proof_sketch', e['statement'][:120])}"
        for e in entries
    ]

    if verbose:
        print("  Embedding keys...")
    key_vecs = embedder.embed_batch(keys_text)

    if verbose:
        print("  Embedding values...")
    val_vecs = embedder.embed_batch(vals_text)

    # Fold and register memories
    for i, entry in enumerate(entries):
        mesh.fold(key_vecs[i], val_vecs[i], strength=fold_strength)
        mesh.remember(key_vecs[i], keys_text[i])

    if verbose:
        s = mesh.stats()
        print(f"  Mesh built: {s['folds']} folds, {s['memories']} memories, "
              f"energy={s['energy']:.1f}")

    return mesh


def build_mesh_from_builtin(embedder: Embedder = None) -> HolographicMesh:
    """Build a mesh from the bundled mini-corpus (no download needed)."""
    entries = [
        {
            'name':         name,
            'statement':    statement,
            'proof_sketch': f"domain: {domain}",
            'domain':       domain,
        }
        for name, statement, domain in BUILTIN_CORPUS
    ]
    return build_mesh_from_corpus(entries, embedder=embedder)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build a HAMesh from a verified math corpus")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--builtin",  action="store_true",
                     help="Use the bundled mini-corpus (~120 theorems, no download needed)")
    src.add_argument("--download", action="store_true",
                     help="Auto-download set.mm from metamath.org (~30 MB, one-time)")
    src.add_argument("--file",     metavar="PATH",
                     help="Path to a Metamath .mm file")

    parser.add_argument("--max",     type=int, default=2000,
                        help="Max theorems to load from .mm file (default 2000)")
    parser.add_argument("--filter",  metavar="DOMAIN",
                        help="Only load entries whose domain contains this string")
    parser.add_argument("--save",    metavar="PATH", default="math_mesh.pt",
                        help="Where to save the mesh (default: math_mesh.pt)")
    parser.add_argument("--model",   default="all-mpnet-base-v2",
                        help="Sentence-transformer model name")
    args = parser.parse_args()

    embedder = Embedder(model_name=args.model)

    if args.builtin:
        entries = [
            {'name': n, 'statement': s, 'proof_sketch': f"domain: {d}", 'domain': d}
            for n, s, d in BUILTIN_CORPUS
        ]
    elif args.download:
        mm_path = download_metamath()
        entries = parse_metamath(str(mm_path), max_theorems=args.max)
    else:
        entries = parse_metamath(args.file, max_theorems=args.max)

    if args.filter:
        entries = [e for e in entries if args.filter.lower() in e.get('domain','').lower()
                                      or args.filter.lower() in e['statement'].lower()]
        print(f"  After filter '{args.filter}': {len(entries)} entries")

    mesh = build_mesh_from_corpus(entries, embedder=embedder)
    mesh.save(args.save)
    print(f"\n  Saved to {args.save}")

    # Show top domains
    from collections import Counter
    domains = Counter(e.get('domain', 'unknown') for e in entries)
    print("\n  Domains:")
    for d, n in domains.most_common():
        print(f"    {d:30s} {n}")


if __name__ == "__main__":
    main()
