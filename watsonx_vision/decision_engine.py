"""
Decision Engine Module

Multi-criteria decision engine that combines validation results
to produce final decisions with weighted scoring.

Extracted from IBM Watsonx Loan Preprocessing Agents project.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, date

from .fraud_detector import FraudResult, FraudSeverity
from .cross_validator import ValidationResult, InconsistencySeverity


class DecisionStatus(Enum):
    """Possible decision outcomes"""
    APPROVED = "approved"
    REJECTED = "rejected"
    PENDING_REVIEW = "pending_review"
    NEEDS_MORE_INFO = "needs_more_info"


@dataclass
class CriterionResult:
    """
    Result of evaluating a single criterion.

    Attributes:
        name: Criterion name
        passed: Whether this criterion passed
        score: Score for this criterion (0-100)
        weight: Weight of this criterion in final decision
        details: Additional details or explanation
    """
    name: str
    passed: bool
    score: int
    weight: float
    details: str

    def weighted_score(self) -> float:
        """Calculate weighted score"""
        return self.score * self.weight


@dataclass
class Decision:
    """
    Final decision from the decision engine.

    Attributes:
        status: The decision outcome (approved, rejected, etc.)
        overall_score: Weighted average of all criteria (0-100)
        criteria_results: List of individual criterion results
        reasons: List of reasons for the decision
        recommendations: List of recommended actions
        timestamp: When the decision was made
        metadata: Additional metadata about the decision
    """
    status: DecisionStatus
    overall_score: float
    criteria_results: List[CriterionResult]
    reasons: List[str]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return {
            "status": self.status.value,
            "overall_score": round(self.overall_score, 2),
            "criteria_results": [
                {
                    "name": cr.name,
                    "passed": cr.passed,
                    "score": cr.score,
                    "weight": cr.weight,
                    "weighted_score": round(cr.weighted_score(), 2),
                    "details": cr.details
                }
                for cr in self.criteria_results
            ],
            "reasons": self.reasons,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

    def summary(self) -> str:
        """Generate a one-line summary"""
        emoji = {
            DecisionStatus.APPROVED: "✅",
            DecisionStatus.REJECTED: "❌",
            DecisionStatus.PENDING_REVIEW: "⏳",
            DecisionStatus.NEEDS_MORE_INFO: "📋"
        }.get(self.status, "❓")

        return f"{emoji} {self.status.value.upper()} (Score: {self.overall_score:.1f}/100)"


class DecisionEngine:
    """
    Multi-criteria decision engine with weighted scoring.

    Combines multiple validation results (fraud detection, cross-validation,
    custom criteria) into a final decision using configurable weights and thresholds.

    Example:
        >>> from watsonx_vision import DecisionEngine, DecisionStatus
        >>> engine = DecisionEngine(
        ...     approval_threshold=75.0,
        ...     rejection_threshold=40.0
        ... )
        >>> # Add custom criteria
        >>> engine.add_criterion(
        ...     "age_verification",
        ...     weight=0.1,
        ...     evaluator=lambda data: data.get("age", 0) >= 18
        ... )
        >>> # Make decision
        >>> decision = engine.decide(
        ...     fraud_results=[fraud_result1, fraud_result2],
        ...     validation_result=cross_validation_result,
        ...     custom_data={"age": 25}
        ... )
        >>> print(decision.summary())
    """

    def __init__(
        self,
        approval_threshold: float = 75.0,
        rejection_threshold: float = 40.0,
        fraud_weight: float = 0.4,
        cross_validation_weight: float = 0.3,
        custom_criteria_weight: float = 0.3
    ):
        """
        Initialize decision engine.

        Args:
            approval_threshold: Minimum score for approval (0-100)
            rejection_threshold: Score below which rejection occurs (0-100)
            fraud_weight: Weight for fraud detection results
            cross_validation_weight: Weight for cross-validation results
            custom_criteria_weight: Weight for custom criteria (distributed among them)
        """
        self.approval_threshold = approval_threshold
        self.rejection_threshold = rejection_threshold
        self.fraud_weight = fraud_weight
        self.cross_validation_weight = cross_validation_weight
        self.custom_criteria_weight = custom_criteria_weight

        # Custom criteria: name -> (weight_fraction, evaluator_function)
        self._custom_criteria: Dict[str, tuple] = {}

    def add_criterion(
        self,
        name: str,
        weight: float,
        evaluator: Callable[[Dict], bool],
        score_on_pass: int = 100,
        score_on_fail: int = 0
    ):
        """
        Add a custom criterion to the decision engine.

        Args:
            name: Unique name for this criterion
            weight: Relative weight (will be normalized)
            evaluator: Function that takes custom_data dict and returns bool
            score_on_pass: Score when criterion passes (default 100)
            score_on_fail: Score when criterion fails (default 0)

        Example:
            >>> engine.add_criterion(
            ...     "minimum_income",
            ...     weight=0.5,
            ...     evaluator=lambda d: d.get("annual_income", 0) >= 50000
            ... )
        """
        self._custom_criteria[name] = (weight, evaluator, score_on_pass, score_on_fail)

    def remove_criterion(self, name: str):
        """Remove a custom criterion by name"""
        if name in self._custom_criteria:
            del self._custom_criteria[name]

    def decide(
        self,
        fraud_results: Optional[List[FraudResult]] = None,
        validation_result: Optional[ValidationResult] = None,
        custom_data: Optional[Dict[str, Any]] = None,
        override_status: Optional[DecisionStatus] = None
    ) -> Decision:
        """
        Make a decision based on all inputs.

        Args:
            fraud_results: List of fraud detection results for documents
            validation_result: Cross-validation result
            custom_data: Data for custom criterion evaluation
            override_status: Force a specific status (for manual overrides)

        Returns:
            Decision object with final result

        Example:
            >>> decision = engine.decide(
            ...     fraud_results=[passport_fraud, license_fraud],
            ...     validation_result=cross_val_result,
            ...     custom_data={"age": 25, "annual_income": 75000}
            ... )
        """
        criteria_results = []
        reasons = []
        recommendations = []

        # Evaluate fraud detection
        if fraud_results:
            fraud_criterion = self._evaluate_fraud(fraud_results)
            criteria_results.append(fraud_criterion)

            if not fraud_criterion.passed:
                reasons.append(f"Document fraud detection failed: {fraud_criterion.details}")
                recommendations.append("Request original documents for manual verification")

        # Evaluate cross-validation
        if validation_result:
            cv_criterion = self._evaluate_cross_validation(validation_result)
            criteria_results.append(cv_criterion)

            if not cv_criterion.passed:
                reasons.append(f"Cross-validation failed: {cv_criterion.details}")
                recommendations.append("Review and resolve data inconsistencies")

        # Evaluate custom criteria
        if custom_data and self._custom_criteria:
            custom_results = self._evaluate_custom_criteria(custom_data)
            criteria_results.extend(custom_results)

            for cr in custom_results:
                if not cr.passed:
                    reasons.append(f"{cr.name} check failed: {cr.details}")

        # Calculate overall score
        if criteria_results:
            total_weight = sum(cr.weight for cr in criteria_results)
            overall_score = sum(cr.weighted_score() for cr in criteria_results) / total_weight if total_weight > 0 else 0
        else:
            overall_score = 0

        # Determine status
        if override_status:
            status = override_status
        else:
            status = self._determine_status(overall_score, criteria_results)

        # Add status-specific recommendations
        if status == DecisionStatus.APPROVED:
            recommendations.append("Proceed with processing")
        elif status == DecisionStatus.REJECTED:
            recommendations.append("Document the rejection reason and notify applicant")
        elif status == DecisionStatus.PENDING_REVIEW:
            recommendations.append("Escalate to senior reviewer for manual assessment")
        elif status == DecisionStatus.NEEDS_MORE_INFO:
            recommendations.append("Request additional documentation from applicant")

        return Decision(
            status=status,
            overall_score=overall_score,
            criteria_results=criteria_results,
            reasons=reasons if reasons else ["All criteria passed"],
            recommendations=recommendations,
            metadata={
                "fraud_results_count": len(fraud_results) if fraud_results else 0,
                "has_cross_validation": validation_result is not None,
                "custom_criteria_count": len(self._custom_criteria)
            }
        )

    def _evaluate_fraud(self, fraud_results: List[FraudResult]) -> CriterionResult:
        """Evaluate fraud detection results"""
        if not fraud_results:
            return CriterionResult(
                name="document_authenticity",
                passed=True,
                score=100,
                weight=self.fraud_weight,
                details="No documents to validate"
            )

        valid_count = sum(1 for r in fraud_results if r.valid)
        total_count = len(fraud_results)
        pass_rate = valid_count / total_count if total_count > 0 else 0

        # Check for critical severity
        has_critical = any(r.severity == FraudSeverity.CRITICAL for r in fraud_results)
        has_high = any(r.severity == FraudSeverity.HIGH for r in fraud_results)

        # Calculate score
        base_score = int(pass_rate * 100)

        # Penalize for severity
        if has_critical:
            score = min(base_score, 20)
        elif has_high:
            score = min(base_score, 50)
        else:
            score = base_score

        passed = valid_count == total_count and not has_critical

        # Get reasons from failed documents
        failed_docs = [r for r in fraud_results if not r.valid]
        if failed_docs:
            details = f"{valid_count}/{total_count} documents validated. Issues: " + \
                      ", ".join([f"{r.filename or 'Unknown'}: {r.reason[:50]}" for r in failed_docs[:3]])
        else:
            details = f"All {total_count} documents validated as authentic"

        return CriterionResult(
            name="document_authenticity",
            passed=passed,
            score=score,
            weight=self.fraud_weight,
            details=details
        )

    def _evaluate_cross_validation(self, result: ValidationResult) -> CriterionResult:
        """Evaluate cross-validation results"""
        # Check severity of inconsistencies
        critical_count = sum(1 for i in result.inconsistencies if i.severity == InconsistencySeverity.CRITICAL)
        high_count = sum(1 for i in result.inconsistencies if i.severity == InconsistencySeverity.HIGH)

        # Calculate score based on confidence and inconsistencies
        base_score = result.confidence

        # Penalize for inconsistencies
        penalty = (critical_count * 30) + (high_count * 15) + (result.total_inconsistencies * 5)
        score = max(0, min(100, base_score - penalty))

        passed = result.passed and critical_count == 0

        details = result.summary or f"{result.total_inconsistencies} inconsistencies found"

        return CriterionResult(
            name="cross_validation",
            passed=passed,
            score=score,
            weight=self.cross_validation_weight,
            details=details
        )

    def _evaluate_custom_criteria(self, data: Dict) -> List[CriterionResult]:
        """Evaluate all custom criteria"""
        results = []

        # Calculate normalized weights
        total_custom_weight = sum(c[0] for c in self._custom_criteria.values())

        for name, (weight, evaluator, score_pass, score_fail) in self._custom_criteria.items():
            try:
                passed = evaluator(data)
                score = score_pass if passed else score_fail

                # Normalize weight to custom_criteria_weight portion
                normalized_weight = (weight / total_custom_weight * self.custom_criteria_weight) if total_custom_weight > 0 else 0

                results.append(CriterionResult(
                    name=name,
                    passed=passed,
                    score=score,
                    weight=normalized_weight,
                    details=f"{'Passed' if passed else 'Failed'}"
                ))
            except Exception as e:
                results.append(CriterionResult(
                    name=name,
                    passed=False,
                    score=0,
                    weight=weight,
                    details=f"Error evaluating criterion: {str(e)}"
                ))

        return results

    def _determine_status(
        self,
        overall_score: float,
        criteria_results: List[CriterionResult]
    ) -> DecisionStatus:
        """Determine final status based on score and criteria"""

        # Check for any critical failures
        has_critical_failure = any(
            not cr.passed and cr.score < 20
            for cr in criteria_results
        )

        if has_critical_failure:
            return DecisionStatus.REJECTED

        # Score-based decision
        if overall_score >= self.approval_threshold:
            return DecisionStatus.APPROVED
        elif overall_score <= self.rejection_threshold:
            return DecisionStatus.REJECTED
        elif overall_score >= (self.approval_threshold + self.rejection_threshold) / 2:
            return DecisionStatus.PENDING_REVIEW
        else:
            return DecisionStatus.NEEDS_MORE_INFO


class LoanDecisionEngine(DecisionEngine):
    """
    Specialized decision engine for loan applications.

    Pre-configured with common loan criteria:
    - Age verification (18+)
    - Income requirements
    - Debt-to-income ratio
    - Employment verification
    """

    def __init__(
        self,
        min_age: int = 18,
        min_income: float = 0,
        max_dti: float = 0.43,  # Standard DTI threshold
        **kwargs
    ):
        """
        Initialize loan decision engine.

        Args:
            min_age: Minimum applicant age
            min_income: Minimum annual income required
            max_dti: Maximum debt-to-income ratio allowed
            **kwargs: Additional arguments for parent DecisionEngine
        """
        super().__init__(**kwargs)

        self.min_age = min_age
        self.min_income = min_income
        self.max_dti = max_dti

        # Add standard loan criteria
        self.add_criterion(
            "age_verification",
            weight=0.2,
            evaluator=self._check_age,
            score_on_pass=100,
            score_on_fail=0
        )

        if min_income > 0:
            self.add_criterion(
                "income_requirement",
                weight=0.3,
                evaluator=self._check_income,
                score_on_pass=100,
                score_on_fail=0
            )

        self.add_criterion(
            "dti_ratio",
            weight=0.3,
            evaluator=self._check_dti,
            score_on_pass=100,
            score_on_fail=30
        )

    def _check_age(self, data: Dict) -> bool:
        """Check if applicant meets age requirement"""
        # Try various age-related fields
        age = data.get("age") or data.get("applicant_age")

        if age is None:
            # Try to calculate from DOB
            dob = data.get("dob") or data.get("date_of_birth")
            if dob:
                try:
                    if isinstance(dob, str):
                        # Try common date formats
                        for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"]:
                            try:
                                dob_date = datetime.strptime(dob, fmt).date()
                                break
                            except ValueError:
                                continue
                        else:
                            return True  # Can't parse, assume valid
                    elif isinstance(dob, (date, datetime)):
                        dob_date = dob if isinstance(dob, date) else dob.date()
                    else:
                        return True

                    today = date.today()
                    age = today.year - dob_date.year - (
                        (today.month, today.day) < (dob_date.month, dob_date.day)
                    )
                except Exception:
                    return True  # Error parsing, assume valid

        if age is None:
            return True  # No age info, assume valid

        return int(age) >= self.min_age

    def _check_income(self, data: Dict) -> bool:
        """Check if applicant meets income requirement"""
        income = (
            data.get("annual_income") or
            data.get("gross_income") or
            data.get("yearly_income") or
            0
        )

        # Convert monthly to annual if needed
        monthly = data.get("monthly_income")
        if monthly and not income:
            income = float(monthly) * 12

        return float(income) >= self.min_income

    def _check_dti(self, data: Dict) -> bool:
        """Check debt-to-income ratio"""
        dti = data.get("dti") or data.get("debt_to_income_ratio")

        if dti is None:
            # Try to calculate
            debt = data.get("monthly_debt") or data.get("total_monthly_debt") or 0
            income = data.get("monthly_income") or data.get("gross_monthly_income") or 0

            if income > 0:
                dti = float(debt) / float(income)
            else:
                return True  # Can't calculate, assume valid

        return float(dti) <= self.max_dti

    def calculate_dti(self, monthly_debt: float, monthly_income: float) -> float:
        """Helper to calculate DTI ratio"""
        if monthly_income <= 0:
            return float('inf')
        return monthly_debt / monthly_income
