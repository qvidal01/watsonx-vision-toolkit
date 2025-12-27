"""Tests for DecisionEngine module"""

import pytest
from datetime import datetime
from watsonx_vision.decision_engine import (
    DecisionEngine,
    LoanDecisionEngine,
    Decision,
    DecisionStatus,
    CriterionResult
)
from watsonx_vision.fraud_detector import FraudResult, FraudSeverity
from watsonx_vision.cross_validator import (
    ValidationResult,
    Inconsistency,
    InconsistencySeverity
)


class TestCriterionResult:
    """Tests for CriterionResult dataclass"""

    def test_weighted_score(self):
        """Test weighted score calculation"""
        result = CriterionResult(
            name="test_criterion",
            passed=True,
            score=80,
            weight=0.5,
            details="Test passed"
        )

        assert result.weighted_score() == 40.0


class TestDecision:
    """Tests for Decision dataclass"""

    def test_decision_creation(self):
        """Test creating a Decision"""
        decision = Decision(
            status=DecisionStatus.APPROVED,
            overall_score=85.0,
            criteria_results=[],
            reasons=["All criteria passed"],
            recommendations=["Proceed with processing"]
        )

        assert decision.status == DecisionStatus.APPROVED
        assert decision.overall_score == 85.0

    def test_decision_to_dict(self):
        """Test converting Decision to dictionary"""
        decision = Decision(
            status=DecisionStatus.REJECTED,
            overall_score=35.0,
            criteria_results=[
                CriterionResult("fraud", False, 20, 0.4, "Failed")
            ],
            reasons=["Fraud detected"],
            recommendations=["Reject application"]
        )

        result = decision.to_dict()

        assert result["status"] == "rejected"
        assert result["overall_score"] == 35.0
        assert len(result["criteria_results"]) == 1

    def test_decision_summary(self):
        """Test decision summary generation"""
        decision = Decision(
            status=DecisionStatus.APPROVED,
            overall_score=90.0,
            criteria_results=[],
            reasons=[],
            recommendations=[]
        )

        summary = decision.summary()
        assert "APPROVED" in summary
        assert "90.0" in summary


class TestDecisionEngine:
    """Tests for DecisionEngine class"""

    def test_engine_initialization(self):
        """Test initializing DecisionEngine"""
        engine = DecisionEngine(
            approval_threshold=80.0,
            rejection_threshold=30.0
        )

        assert engine.approval_threshold == 80.0
        assert engine.rejection_threshold == 30.0

    def test_add_criterion(self):
        """Test adding custom criterion"""
        engine = DecisionEngine()

        engine.add_criterion(
            "test_criterion",
            weight=0.5,
            evaluator=lambda d: d.get("value", 0) > 100
        )

        assert "test_criterion" in engine._custom_criteria

    def test_remove_criterion(self):
        """Test removing custom criterion"""
        engine = DecisionEngine()
        engine.add_criterion("temp", weight=0.5, evaluator=lambda d: True)
        engine.remove_criterion("temp")

        assert "temp" not in engine._custom_criteria

    def test_decide_with_passing_criteria(self):
        """Test decision with all passing criteria"""
        engine = DecisionEngine()

        # Create passing fraud results
        fraud_results = [
            FraudResult(True, 90, "OK", 95, 85, [], FraudSeverity.NONE, "doc1.png"),
            FraudResult(True, 85, "OK", 90, 80, [], FraudSeverity.NONE, "doc2.png"),
        ]

        # Create passing validation result
        validation_result = ValidationResult(
            passed=True,
            total_inconsistencies=0,
            inconsistencies=[],
            matched_fields=["name", "dob"],
            summary="All data consistent",
            confidence=95
        )

        decision = engine.decide(
            fraud_results=fraud_results,
            validation_result=validation_result
        )

        assert decision.status == DecisionStatus.APPROVED
        assert decision.overall_score >= 75

    def test_decide_with_failing_fraud(self):
        """Test decision with failing fraud detection"""
        engine = DecisionEngine()

        fraud_results = [
            FraudResult(False, 20, "Fake", 10, 30, ["Forgery"], FraudSeverity.CRITICAL, "fake.png"),
        ]

        decision = engine.decide(fraud_results=fraud_results)

        assert decision.status == DecisionStatus.REJECTED

    def test_decide_with_failing_cross_validation(self):
        """Test decision with failing cross-validation"""
        engine = DecisionEngine(
            approval_threshold=75.0,
            rejection_threshold=40.0
        )

        validation_result = ValidationResult(
            passed=False,
            total_inconsistencies=2,
            inconsistencies=[
                Inconsistency(
                    "ssn", "123-45-6789", "987-65-4321",
                    "Application", "Tax Return",
                    InconsistencySeverity.CRITICAL,
                    "SSN mismatch"
                ),
            ],
            matched_fields=["name"],
            summary="Critical inconsistency in SSN",
            confidence=30
        )

        decision = engine.decide(validation_result=validation_result)

        assert decision.status in [DecisionStatus.REJECTED, DecisionStatus.NEEDS_MORE_INFO]

    def test_decide_with_custom_criteria(self):
        """Test decision with custom criteria"""
        engine = DecisionEngine(custom_criteria_weight=0.5)

        engine.add_criterion(
            "minimum_income",
            weight=1.0,
            evaluator=lambda d: d.get("income", 0) >= 50000
        )

        # Test passing custom criteria
        decision = engine.decide(custom_data={"income": 75000})
        assert any(cr.name == "minimum_income" and cr.passed for cr in decision.criteria_results)

        # Test failing custom criteria
        decision = engine.decide(custom_data={"income": 25000})
        assert any(cr.name == "minimum_income" and not cr.passed for cr in decision.criteria_results)


class TestLoanDecisionEngine:
    """Tests for LoanDecisionEngine class"""

    def test_loan_engine_initialization(self):
        """Test initializing LoanDecisionEngine"""
        engine = LoanDecisionEngine(
            min_age=21,
            min_income=40000,
            max_dti=0.40
        )

        assert engine.min_age == 21
        assert engine.min_income == 40000
        assert engine.max_dti == 0.40
        assert "age_verification" in engine._custom_criteria

    def test_age_verification_pass(self):
        """Test age verification passing"""
        engine = LoanDecisionEngine(min_age=18)

        decision = engine.decide(custom_data={"age": 25})

        age_result = next(
            (cr for cr in decision.criteria_results if cr.name == "age_verification"),
            None
        )
        assert age_result is not None
        assert age_result.passed is True

    def test_age_verification_fail(self):
        """Test age verification failing"""
        engine = LoanDecisionEngine(min_age=18)

        decision = engine.decide(custom_data={"age": 16})

        age_result = next(
            (cr for cr in decision.criteria_results if cr.name == "age_verification"),
            None
        )
        assert age_result is not None
        assert age_result.passed is False

    def test_dti_calculation(self):
        """Test DTI ratio calculation"""
        engine = LoanDecisionEngine(max_dti=0.43)

        decision = engine.decide(custom_data={
            "monthly_debt": 1500,
            "monthly_income": 5000,  # DTI = 0.30
            "age": 30
        })

        dti_result = next(
            (cr for cr in decision.criteria_results if cr.name == "dti_ratio"),
            None
        )
        assert dti_result is not None
        assert dti_result.passed is True

    def test_calculate_dti_helper(self):
        """Test DTI calculation helper"""
        engine = LoanDecisionEngine()

        assert engine.calculate_dti(1500, 5000) == 0.3
        assert engine.calculate_dti(2150, 5000) == 0.43
        assert engine.calculate_dti(0, 0) == float('inf')
