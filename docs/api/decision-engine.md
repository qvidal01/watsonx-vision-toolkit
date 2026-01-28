# DecisionEngine

Multi-criteria weighted decision making engine.

## DecisionEngine

### Constructor

```python
from watsonx_vision import DecisionEngine

engine = DecisionEngine(
    approval_threshold=80,
    rejection_threshold=40
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `approval_threshold` | `int` | `80` | Score threshold for approval |
| `rejection_threshold` | `int` | `40` | Score threshold for rejection |

---

### add_criterion

Add a custom evaluation criterion.

```python
engine.add_criterion(
    name="income_check",
    weight=30,
    evaluator=lambda data: data.get("income", 0) >= 50000
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Criterion identifier |
| `weight` | `int` | Weight (contribution to total score) |
| `evaluator` | `Callable[[Dict], bool]` | Function returning True/False |

---

### remove_criterion

Remove a criterion by name.

```python
engine.remove_criterion("income_check")
```

---

### decide

Make a decision based on all criteria.

```python
decision = engine.decide(
    application_data={"name": "John", "income": 75000, "age": 35},
    fraud_result=fraud_result,      # Optional FraudResult
    validation_result=validation_result  # Optional ValidationResult
)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `application_data` | `Dict` | Yes | Application data for evaluators |
| `fraud_result` | `FraudResult \| None` | No | Fraud detection result |
| `validation_result` | `ValidationResult \| None` | No | Cross-validation result |

**Returns:** `Decision`

---

## LoanDecisionEngine

Pre-configured decision engine for loan applications.

```python
from watsonx_vision.decision_engine import LoanDecisionEngine

engine = LoanDecisionEngine(
    min_age=18,
    min_income=30000,
    max_dti=0.43
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_age` | `int` | `18` | Minimum applicant age |
| `min_income` | `int` | `30000` | Minimum annual income |
| `max_dti` | `float` | `0.43` | Maximum debt-to-income ratio |

### Built-in Criteria

| Criterion | Weight | Condition |
|-----------|--------|-----------|
| Age verification | 20 | age ≥ min_age |
| Income requirement | 30 | annual_income ≥ min_income |
| Debt-to-income ratio | 25 | DTI ≤ max_dti |
| Fraud check | 15 | fraud_result.valid |
| Cross-validation | 10 | validation_result.passed |

### Usage

```python
decision = engine.decide(
    application_data={
        "dob": "1990-01-15",  # For age calculation
        "annual_income": 75000,
        "monthly_debt": 1500
    },
    fraud_result=fraud_result,
    validation_result=validation_result
)
```

---

## Decision

Dataclass for decision results.

```python
@dataclass
class Decision:
    status: DecisionStatus      # APPROVED, REJECTED, etc.
    score: int                  # 0-100 overall score
    criteria_results: List[CriterionResult]
    reasons: List[str]          # Explanation list
    timestamp: datetime
```

### Methods

#### to_dict

Convert to dictionary.

```python
data = decision.to_dict()
```

#### summary

Get human-readable summary.

```python
print(decision.summary())
# Output: "✅ APPROVED (Score: 85/100)"
```

---

## CriterionResult

Dataclass for individual criterion results.

```python
@dataclass
class CriterionResult:
    name: str           # Criterion name
    passed: bool        # Whether criterion passed
    weight: int         # Criterion weight
    contribution: int   # Score contribution (weight if passed, 0 if failed)
```

---

## DecisionStatus Enum

```python
from watsonx_vision.decision_engine import DecisionStatus

class DecisionStatus(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    PENDING_REVIEW = "pending_review"
    NEEDS_MORE_INFO = "needs_more_info"
```

### Status Determination

| Condition | Status |
|-----------|--------|
| score ≥ approval_threshold | `APPROVED` |
| score < rejection_threshold | `REJECTED` |
| Otherwise | `PENDING_REVIEW` |

---

## Example

```python
from watsonx_vision import DecisionEngine
from watsonx_vision.decision_engine import LoanDecisionEngine

# Basic decision engine
engine = DecisionEngine(
    approval_threshold=75,
    rejection_threshold=40
)

# Add custom criteria
engine.add_criterion(
    "age_check",
    weight=20,
    evaluator=lambda d: d.get("age", 0) >= 18
)

engine.add_criterion(
    "income_check",
    weight=30,
    evaluator=lambda d: d.get("annual_income", 0) >= 50000
)

engine.add_criterion(
    "credit_score",
    weight=25,
    evaluator=lambda d: d.get("credit_score", 0) >= 650
)

engine.add_criterion(
    "employment",
    weight=25,
    evaluator=lambda d: d.get("employment_years", 0) >= 2
)

# Make decision
decision = engine.decide(
    application_data={
        "age": 35,
        "annual_income": 75000,
        "credit_score": 720,
        "employment_years": 5
    }
)

print(decision.summary())
# ✅ APPROVED (Score: 100/100)

print(f"Status: {decision.status.value}")
print(f"Score: {decision.score}")
print(f"Reasons: {decision.reasons}")

for cr in decision.criteria_results:
    status = "✓" if cr.passed else "✗"
    print(f"  {status} {cr.name}: {cr.contribution}/{cr.weight}")

# Using LoanDecisionEngine
loan_engine = LoanDecisionEngine(min_age=18, min_income=30000, max_dti=0.43)

decision = loan_engine.decide(
    application_data={
        "dob": "1990-01-15",
        "annual_income": 85000,
        "monthly_debt": 1500
    },
    fraud_result=fraud_result,
    validation_result=validation_result
)
```
