"""
Enhanced Error Handling and User Feedback System for HouseBrain

This module provides comprehensive error handling with:
- Detailed error categorization and reporting
- User-friendly error messages with solutions
- Error recovery suggestions
- Performance monitoring and alerts
- Validation feedback with specific guidance
- Progressive error disclosure
- Context-aware help system
"""

from __future__ import annotations

import traceback
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels"""
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SUCCESS = "success"


class ErrorCategory(Enum):
    """Error categories for better organization"""
    VALIDATION = "validation"
    FILE_IO = "file_io"
    GEOMETRY = "geometry"
    MATERIAL = "material"
    RENDERING = "rendering"
    EXPORT = "export"
    IMPORT = "import"
    PERFORMANCE = "performance"
    USER_INPUT = "user_input"
    SYSTEM = "system"


@dataclass
class ErrorDetails:
    """Detailed error information"""
    id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    technical_details: str
    user_message: str
    suggestions: List[str]
    context: Dict[str, Any]
    timestamp: float
    stack_trace: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "technical_details": self.technical_details,
            "user_message": self.user_message,
            "suggestions": self.suggestions,
            "context": self.context,
            "timestamp": self.timestamp,
            "stack_trace": self.stack_trace
        }


@dataclass
class ValidationResult:
    """Result of validation operation"""
    is_valid: bool
    errors: List[ErrorDetails]
    warnings: List[ErrorDetails]
    success_messages: List[str]
    performance_metrics: Dict[str, Any]
    
    def has_critical_errors(self) -> bool:
        return any(error.severity == ErrorSeverity.CRITICAL for error in self.errors)
    
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    def get_error_summary(self) -> str:
        if not self.errors and not self.warnings:
            return "All validations passed successfully"
        
        error_count = len(self.errors)
        warning_count = len(self.warnings)
        
        summary = []
        if error_count > 0:
            summary.append(f"{error_count} error{'s' if error_count != 1 else ''}")
        if warning_count > 0:
            summary.append(f"{warning_count} warning{'s' if warning_count != 1 else ''}")
        
        return " and ".join(summary) + " found"


class EnhancedErrorHandler:
    """Enhanced error handling and user feedback system"""
    
    def __init__(self, log_file: str = "housebrain_errors.log"):
        self.error_history = []
        self.error_patterns = {}
        self.user_context = {}
        self.performance_thresholds = self._initialize_performance_thresholds()
        self.error_solutions = self._initialize_error_solutions()
        self.validation_rules = self._initialize_validation_rules()
        
        # Setup logging
        self.logger = self._setup_logging(log_file)
        
        print("🛡️ Enhanced Error Handler Initialized")
    
    def handle_error(
        self,
        exception: Exception,
        category: ErrorCategory,
        context: Dict[str, Any] = None,
        user_action: str = None
    ) -> ErrorDetails:
        """Handle an exception with detailed error processing"""
        
        error_id = self._generate_error_id()
        
        # Determine severity
        severity = self._determine_severity(exception, category)
        
        # Create detailed error
        error_details = ErrorDetails(
            id=error_id,
            category=category,
            severity=severity,
            message=str(exception),
            technical_details=self._extract_technical_details(exception),
            user_message=self._create_user_friendly_message(exception, category, user_action),
            suggestions=self._generate_suggestions(exception, category, context),
            context=context or {},
            timestamp=time.time(),
            stack_trace=traceback.format_exc()
        )
        
        # Log error
        self._log_error(error_details)
        
        # Store in history
        self.error_history.append(error_details)
        
        # Update error patterns
        self._update_error_patterns(error_details)
        
        # Check for critical errors that need immediate attention
        if severity == ErrorSeverity.CRITICAL:
            self._handle_critical_error(error_details)
        
        return error_details
    
    def validate_input(
        self,
        data: Any,
        validation_type: str,
        context: Dict[str, Any] = None
    ) -> ValidationResult:
        """Comprehensive input validation with detailed feedback"""
        
        start_time = time.time()
        errors = []
        warnings = []
        success_messages = []
        
        # Get validation rules for this type
        rules = self.validation_rules.get(validation_type, [])
        
        for rule in rules:
            try:
                result = self._apply_validation_rule(data, rule, context)
                
                if result["status"] == "error":
                    error = self._create_validation_error(rule, result, data, context)
                    errors.append(error)
                elif result["status"] == "warning":
                    warning = self._create_validation_warning(rule, result, data, context)
                    warnings.append(warning)
                elif result["status"] == "success":
                    success_messages.append(result.get("message", f"{rule['name']} validation passed"))
                
            except Exception as e:
                # Validation rule itself failed
                error = self.handle_error(e, ErrorCategory.VALIDATION, {
                    "rule": rule["name"],
                    "validation_type": validation_type
                })
                errors.append(error)
        
        validation_time = time.time() - start_time
        
        # Check performance
        performance_metrics = {
            "validation_time": validation_time,
            "rules_applied": len(rules),
            "errors_found": len(errors),
            "warnings_found": len(warnings)
        }
        
        if validation_time > self.performance_thresholds["validation_time"]:
            warning = self._create_performance_warning(
                "validation_performance",
                f"Validation took {validation_time:.2f}s (threshold: {self.performance_thresholds['validation_time']}s)",
                {"validation_time": validation_time, "validation_type": validation_type}
            )
            warnings.append(warning)
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            success_messages=success_messages,
            performance_metrics=performance_metrics
        )
    
    def provide_contextual_help(
        self,
        topic: str,
        user_context: Dict[str, Any] = None,
        recent_errors: List[ErrorDetails] = None
    ) -> Dict[str, Any]:
        """Provide contextual help based on user situation"""
        
        help_content = {
            "topic": topic,
            "primary_help": self._get_primary_help(topic),
            "contextual_suggestions": [],
            "related_topics": [],
            "troubleshooting": [],
            "examples": []
        }
        
        # Add contextual suggestions based on recent errors
        if recent_errors:
            help_content["contextual_suggestions"] = self._generate_contextual_suggestions(
                topic, recent_errors
            )
        
        # Add user-specific help based on context
        if user_context:
            help_content["user_specific_help"] = self._generate_user_specific_help(
                topic, user_context
            )
        
        # Add related topics
        help_content["related_topics"] = self._get_related_topics(topic)
        
        # Add troubleshooting for common issues
        help_content["troubleshooting"] = self._get_troubleshooting_steps(topic)
        
        # Add examples
        help_content["examples"] = self._get_help_examples(topic)
        
        return help_content
    
    def monitor_performance(
        self,
        operation_name: str,
        start_time: float,
        end_time: float,
        context: Dict[str, Any] = None
    ) -> Optional[ErrorDetails]:
        """Monitor operation performance and generate warnings if needed"""
        
        duration = end_time - start_time
        threshold = self.performance_thresholds.get(operation_name, 10.0)  # Default 10s
        
        if duration > threshold:
            warning = self._create_performance_warning(
                operation_name,
                f"Operation '{operation_name}' took {duration:.2f}s (threshold: {threshold}s)",
                context or {}
            )
            
            self.error_history.append(warning)
            self._log_error(warning)
            
            return warning
        
        return None
    
    def get_error_analytics(self) -> Dict[str, Any]:
        """Get analytics about error patterns and system health"""
        
        analytics = {
            "total_errors": len(self.error_history),
            "error_by_category": {},
            "error_by_severity": {},
            "recent_errors": [],
            "top_error_patterns": [],
            "system_health": "good",
            "recommendations": []
        }
        
        # Count by category
        for error in self.error_history:
            category = error.category.value
            analytics["error_by_category"][category] = analytics["error_by_category"].get(category, 0) + 1
        
        # Count by severity
        for error in self.error_history:
            severity = error.severity.value
            analytics["error_by_severity"][severity] = analytics["error_by_severity"].get(severity, 0) + 1
        
        # Recent errors (last 10)
        analytics["recent_errors"] = [error.to_dict() for error in self.error_history[-10:]]
        
        # Top error patterns
        analytics["top_error_patterns"] = sorted(
            self.error_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # System health assessment
        critical_errors = analytics["error_by_severity"].get("critical", 0)
        total_errors = len(self.error_history)
        
        if critical_errors > 0:
            analytics["system_health"] = "critical"
        elif total_errors > 50:
            analytics["system_health"] = "degraded"
        elif total_errors > 20:
            analytics["system_health"] = "warning"
        else:
            analytics["system_health"] = "good"
        
        # Generate recommendations
        analytics["recommendations"] = self._generate_system_recommendations(analytics)
        
        return analytics
    
    def generate_user_report(
        self,
        operation: str,
        results: Any,
        errors: List[ErrorDetails] = None,
        warnings: List[ErrorDetails] = None
    ) -> Dict[str, Any]:
        """Generate user-friendly operation report"""
        
        report = {
            "operation": operation,
            "status": "success",
            "summary": "",
            "details": {},
            "next_steps": [],
            "timestamp": time.time()
        }
        
        # Determine overall status
        if errors and any(e.severity == ErrorSeverity.CRITICAL for e in errors):
            report["status"] = "failed"
        elif errors:
            report["status"] = "completed_with_errors"
        elif warnings:
            report["status"] = "completed_with_warnings"
        else:
            report["status"] = "success"
        
        # Generate summary
        report["summary"] = self._generate_operation_summary(operation, results, errors, warnings)
        
        # Add details
        if errors:
            report["details"]["errors"] = [
                {
                    "message": error.user_message,
                    "suggestions": error.suggestions,
                    "severity": error.severity.value
                }
                for error in errors
            ]
        
        if warnings:
            report["details"]["warnings"] = [
                {
                    "message": warning.user_message,
                    "suggestions": warning.suggestions
                }
                for warning in warnings
            ]
        
        # Generate next steps
        report["next_steps"] = self._generate_next_steps(operation, report["status"], errors, warnings)
        
        return report
    
    def create_recovery_plan(
        self,
        errors: List[ErrorDetails],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create a recovery plan for handling multiple errors"""
        
        recovery_plan = {
            "total_errors": len(errors),
            "critical_issues": [],
            "recovery_steps": [],
            "estimated_time": 0,
            "success_probability": 0.9,
            "alternative_approaches": []
        }
        
        # Identify critical issues
        critical_errors = [e for e in errors if e.severity == ErrorSeverity.CRITICAL]
        recovery_plan["critical_issues"] = [
            {
                "category": error.category.value,
                "message": error.user_message,
                "urgency": "high"
            }
            for error in critical_errors
        ]
        
        # Generate recovery steps
        recovery_plan["recovery_steps"] = self._generate_recovery_steps(errors, context)
        
        # Estimate time and success probability
        recovery_plan["estimated_time"] = len(recovery_plan["recovery_steps"]) * 5  # 5 min per step
        recovery_plan["success_probability"] = max(0.5, 1.0 - len(critical_errors) * 0.2)
        
        # Suggest alternative approaches
        if len(critical_errors) > 2:
            recovery_plan["alternative_approaches"] = [
                "Consider simplifying the design requirements",
                "Try a different approach or workflow",
                "Consult the troubleshooting guide",
                "Contact support with error details"
            ]
        
        return recovery_plan
    
    # Helper methods
    
    def _setup_logging(self, log_file: str) -> logging.Logger:
        """Setup error logging"""
        
        logger = logging.getLogger("housebrain_errors")
        logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler  
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _initialize_performance_thresholds(self) -> Dict[str, float]:
        """Initialize performance monitoring thresholds"""
        
        return {
            "validation_time": 2.0,  # seconds
            "geometry_generation": 5.0,
            "material_loading": 1.0,
            "rendering": 30.0,
            "export_operation": 10.0,
            "file_io": 5.0
        }
    
    def _initialize_error_solutions(self) -> Dict[str, Dict]:
        """Initialize common error solutions"""
        
        return {
            "FileNotFoundError": {
                "category": ErrorCategory.FILE_IO,
                "user_message": "The requested file could not be found",
                "suggestions": [
                    "Check that the file path is correct",
                    "Verify the file exists in the specified location", 
                    "Ensure you have permission to access the file",
                    "Try using an absolute file path instead of relative"
                ]
            },
            "ValidationError": {
                "category": ErrorCategory.VALIDATION,
                "user_message": "The provided data does not meet requirements",
                "suggestions": [
                    "Check the input data format and structure",
                    "Verify all required fields are provided",
                    "Ensure data values are within acceptable ranges",
                    "Review the documentation for correct data format"
                ]
            },
            "JSONDecodeError": {
                "category": ErrorCategory.FILE_IO,
                "user_message": "The file contains invalid JSON data",
                "suggestions": [
                    "Check the JSON file for syntax errors",
                    "Verify all brackets and quotes are properly closed",
                    "Use a JSON validator to identify formatting issues",
                    "Ensure the file is not corrupted or truncated"
                ]
            },
            "MemoryError": {
                "category": ErrorCategory.PERFORMANCE,
                "user_message": "The operation requires more memory than available",
                "suggestions": [
                    "Try reducing the complexity of the design",
                    "Close other applications to free up memory",
                    "Process the data in smaller chunks",
                    "Consider upgrading system memory"
                ]
            }
        }
    
    def _initialize_validation_rules(self) -> Dict[str, List[Dict]]:
        """Initialize validation rules for different data types"""
        
        return {
            "house_plan": [
                {
                    "name": "required_fields",
                    "type": "structure",
                    "check": "has_required_fields",
                    "required": ["geometry", "metadata"],
                    "message": "House plan must include geometry and metadata"
                },
                {
                    "name": "geometry_structure",
                    "type": "structure", 
                    "check": "nested_structure",
                    "path": "geometry",
                    "required": ["spaces"],
                    "message": "Geometry must include spaces definition"
                },
                {
                    "name": "space_validity",
                    "type": "geometry",
                    "check": "valid_spaces",
                    "min_area": 1000000,  # 1 m² in mm²
                    "message": "All spaces must have valid areas and boundaries"
                },
                {
                    "name": "material_references",
                    "type": "reference",
                    "check": "valid_material_refs",
                    "message": "All material references must be valid"
                }
            ],
            "material_data": [
                {
                    "name": "required_properties",
                    "type": "structure",
                    "check": "has_required_fields",
                    "required": ["name", "properties"],
                    "message": "Material must have name and properties"
                },
                {
                    "name": "property_values",
                    "type": "values",
                    "check": "valid_property_ranges",
                    "message": "Material properties must be within valid ranges"
                }
            ],
            "component_data": [
                {
                    "name": "component_structure",
                    "type": "structure",
                    "check": "has_required_fields",
                    "required": ["name", "type", "dimensions"],
                    "message": "Component must have name, type, and dimensions"
                },
                {
                    "name": "dimension_validity",
                    "type": "geometry",
                    "check": "valid_dimensions",
                    "min_value": 1,  # mm
                    "max_value": 50000,  # 50m in mm
                    "message": "Component dimensions must be realistic"
                }
            ]
        }
    
    def _generate_error_id(self) -> str:
        """Generate unique error ID"""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _determine_severity(self, exception: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Determine error severity based on exception type and category"""
        
        critical_exceptions = [MemoryError, SystemExit, KeyboardInterrupt]
        error_exceptions = [ValueError, TypeError, FileNotFoundError, KeyError]
        
        if type(exception) in critical_exceptions:
            return ErrorSeverity.CRITICAL
        elif type(exception) in error_exceptions:
            return ErrorSeverity.ERROR
        else:
            return ErrorSeverity.WARNING
    
    def _extract_technical_details(self, exception: Exception) -> str:
        """Extract technical details from exception"""
        
        details = [
            f"Exception Type: {type(exception).__name__}",
            f"Exception Message: {str(exception)}"
        ]
        
        if hasattr(exception, 'args') and exception.args:
            details.append(f"Arguments: {exception.args}")
        
        if hasattr(exception, 'errno'):
            details.append(f"Error Code: {exception.errno}")
        
        if hasattr(exception, 'filename'):
            details.append(f"Filename: {exception.filename}")
        
        return "\n".join(details)
    
    def _create_user_friendly_message(
        self,
        exception: Exception,
        category: ErrorCategory,
        user_action: str = None
    ) -> str:
        """Create user-friendly error message"""
        
        exception_name = type(exception).__name__
        
        if exception_name in self.error_solutions:
            base_message = self.error_solutions[exception_name]["user_message"]
        else:
            base_message = f"An error occurred during {category.value} operation"
        
        if user_action:
            return f"{base_message} while {user_action}"
        else:
            return base_message
    
    def _generate_suggestions(
        self,
        exception: Exception,
        category: ErrorCategory,
        context: Dict[str, Any] = None
    ) -> List[str]:
        """Generate contextual suggestions for error resolution"""
        
        exception_name = type(exception).__name__
        
        if exception_name in self.error_solutions:
            suggestions = self.error_solutions[exception_name]["suggestions"].copy()
        else:
            suggestions = ["Check the error details for more information"]
        
        # Add context-specific suggestions
        if context:
            context_suggestions = self._generate_context_suggestions(exception, category, context)
            suggestions.extend(context_suggestions)
        
        return suggestions
    
    def _generate_context_suggestions(
        self,
        exception: Exception,
        category: ErrorCategory,
        context: Dict[str, Any]
    ) -> List[str]:
        """Generate context-specific suggestions"""
        
        suggestions = []
        
        if category == ErrorCategory.FILE_IO:
            if "file_path" in context:
                suggestions.append(f"Verify the file path: {context['file_path']}")
        
        if category == ErrorCategory.GEOMETRY:
            if "space_count" in context:
                suggestions.append(f"Check the {context['space_count']} spaces in your design")
        
        if category == ErrorCategory.MATERIAL:
            if "material_name" in context:
                suggestions.append(f"Verify material '{context['material_name']}' is available")
        
        return suggestions
    
    def _log_error(self, error_details: ErrorDetails):
        """Log error details"""
        
        log_level = {
            ErrorSeverity.CRITICAL: logging.CRITICAL,
            ErrorSeverity.ERROR: logging.ERROR,
            ErrorSeverity.WARNING: logging.WARNING,
            ErrorSeverity.INFO: logging.INFO,
            ErrorSeverity.SUCCESS: logging.INFO
        }
        
        self.logger.log(
            log_level[error_details.severity],
            f"[{error_details.category.value}] {error_details.message} - {error_details.user_message}"
        )
    
    def _update_error_patterns(self, error_details: ErrorDetails):
        """Update error pattern tracking"""
        
        pattern_key = f"{error_details.category.value}_{type(error_details.message).__name__}"
        self.error_patterns[pattern_key] = self.error_patterns.get(pattern_key, 0) + 1
    
    def _handle_critical_error(self, error_details: ErrorDetails):
        """Handle critical errors that need immediate attention"""
        
        # Log critical error immediately
        self.logger.critical(f"CRITICAL ERROR: {error_details.message}")
        
        # Could trigger additional alerting mechanisms here
        # e.g., send notifications, create backup, etc.
    
    def _apply_validation_rule(
        self,
        data: Any,
        rule: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Apply a single validation rule"""
        
        rule_type = rule.get("type", "unknown")
        rule.get("check", "unknown")
        
        if rule_type == "structure":
            return self._check_structure_rule(data, rule, context)
        elif rule_type == "geometry":
            return self._check_geometry_rule(data, rule, context)
        elif rule_type == "values":
            return self._check_values_rule(data, rule, context)
        elif rule_type == "reference":
            return self._check_reference_rule(data, rule, context)
        else:
            return {"status": "warning", "message": f"Unknown validation rule type: {rule_type}"}
    
    def _check_structure_rule(self, data: Any, rule: Dict, context: Dict) -> Dict[str, Any]:
        """Check structural validation rules"""
        
        if rule.get("check") == "has_required_fields":
            required_fields = rule.get("required", [])
            missing_fields = []
            
            if isinstance(data, dict):
                for field in required_fields:
                    if field not in data:
                        missing_fields.append(field)
            else:
                missing_fields = required_fields
            
            if missing_fields:
                return {
                    "status": "error",
                    "message": f"Missing required fields: {', '.join(missing_fields)}"
                }
            else:
                return {"status": "success", "message": "All required fields present"}
        
        return {"status": "success"}
    
    def _check_geometry_rule(self, data: Any, rule: Dict, context: Dict) -> Dict[str, Any]:
        """Check geometry validation rules"""
        
        if rule.get("check") == "valid_spaces":
            if not isinstance(data, dict) or "geometry" not in data:
                return {"status": "error", "message": "No geometry data found"}
            
            spaces = data.get("geometry", {}).get("spaces", [])
            min_area = rule.get("min_area", 1000000)  # 1 m²
            
            invalid_spaces = []
            for i, space in enumerate(spaces):
                area = space.get("area", 0)
                if area < min_area:
                    invalid_spaces.append(f"Space {i+1}")
            
            if invalid_spaces:
                return {
                    "status": "error",
                    "message": f"Invalid spaces found: {', '.join(invalid_spaces)}"
                }
            else:
                return {"status": "success", "message": f"All {len(spaces)} spaces are valid"}
        
        return {"status": "success"}
    
    def _check_values_rule(self, data: Any, rule: Dict, context: Dict) -> Dict[str, Any]:
        """Check value validation rules"""
        return {"status": "success"}
    
    def _check_reference_rule(self, data: Any, rule: Dict, context: Dict) -> Dict[str, Any]:
        """Check reference validation rules"""
        return {"status": "success"}
    
    def _create_validation_error(
        self,
        rule: Dict,
        result: Dict,
        data: Any,
        context: Dict
    ) -> ErrorDetails:
        """Create validation error details"""
        
        return ErrorDetails(
            id=self._generate_error_id(),
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.ERROR,
            message=result.get("message", "Validation failed"),
            technical_details=f"Rule: {rule['name']}, Check: {rule.get('check', 'unknown')}",
            user_message=rule.get("message", result.get("message", "Validation failed")),
            suggestions=[
                "Review the input data format",
                "Check the documentation for requirements",
                "Verify all required fields are provided"
            ],
            context={"rule": rule["name"], "validation_type": rule.get("type")},
            timestamp=time.time()
        )
    
    def _create_validation_warning(
        self,
        rule: Dict,
        result: Dict,
        data: Any,
        context: Dict
    ) -> ErrorDetails:
        """Create validation warning details"""
        
        return ErrorDetails(
            id=self._generate_error_id(),
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.WARNING,
            message=result.get("message", "Validation warning"),
            technical_details=f"Rule: {rule['name']}",
            user_message=result.get("message", "Validation warning"),
            suggestions=["Review the flagged item", "Consider if this is acceptable"],
            context={"rule": rule["name"]},
            timestamp=time.time()
        )
    
    def _create_performance_warning(
        self,
        operation: str,
        message: str,
        context: Dict[str, Any]
    ) -> ErrorDetails:
        """Create performance warning"""
        
        return ErrorDetails(
            id=self._generate_error_id(),
            category=ErrorCategory.PERFORMANCE,
            severity=ErrorSeverity.WARNING,
            message=message,
            technical_details=f"Performance threshold exceeded for {operation}",
            user_message=f"The {operation} operation is taking longer than expected",
            suggestions=[
                "Try reducing the complexity of the operation",
                "Check system resources and close unnecessary applications",
                "Consider breaking the operation into smaller steps"
            ],
            context=context,
            timestamp=time.time()
        )
    
    # Help and guidance methods (simplified implementations)
    
    def _get_primary_help(self, topic: str) -> str:
        help_topics = {
            "validation": "Validation ensures your design data meets all requirements...",
            "materials": "Materials define the physical properties of building elements...",
            "geometry": "Geometry defines the spatial arrangement of your design...",
            "rendering": "Rendering creates visual representations of your design..."
        }
        return help_topics.get(topic, f"Help for {topic} is not available yet.")
    
    def _generate_contextual_suggestions(self, topic: str, recent_errors: List[ErrorDetails]) -> List[str]:
        suggestions = []
        for error in recent_errors[-3:]:  # Last 3 errors
            if error.category.value in topic.lower():
                suggestions.extend(error.suggestions[:2])  # First 2 suggestions
        return list(set(suggestions))  # Remove duplicates
    
    def _generate_user_specific_help(self, topic: str, user_context: Dict) -> str:
        return f"Based on your current context, here are specific tips for {topic}..."
    
    def _get_related_topics(self, topic: str) -> List[str]:
        related = {
            "validation": ["data_format", "requirements", "troubleshooting"],
            "materials": ["components", "sustainability", "performance"],
            "geometry": ["spaces", "measurements", "validation"]
        }
        return related.get(topic, [])
    
    def _get_troubleshooting_steps(self, topic: str) -> List[str]:
        return [
            f"Step 1: Check {topic} requirements",
            f"Step 2: Verify {topic} data format",
            f"Step 3: Review {topic} examples",
            "Step 4: Contact support if issues persist"
        ]
    
    def _get_help_examples(self, topic: str) -> List[str]:
        return [f"Example 1 for {topic}", f"Example 2 for {topic}"]
    
    def _generate_operation_summary(
        self,
        operation: str,
        results: Any,
        errors: List[ErrorDetails] = None,
        warnings: List[ErrorDetails] = None
    ) -> str:
        """Generate operation summary"""
        
        if not errors and not warnings:
            return f"{operation} completed successfully"
        elif errors:
            return f"{operation} completed with {len(errors)} error(s) and {len(warnings or [])} warning(s)"
        else:
            return f"{operation} completed with {len(warnings)} warning(s)"
    
    def _generate_next_steps(
        self,
        operation: str,
        status: str,
        errors: List[ErrorDetails] = None,
        warnings: List[ErrorDetails] = None
    ) -> List[str]:
        """Generate next steps based on operation results"""
        
        if status == "success":
            return ["Operation completed successfully", "You can proceed to the next step"]
        elif status == "failed":
            return [
                "Review and fix the errors shown above",
                "Retry the operation after making corrections",
                "Contact support if problems persist"
            ]
        else:
            return [
                "Review any warnings to ensure they are acceptable",
                "Consider addressing warnings for better results",
                "Proceed with caution or make improvements"
            ]
    
    def _generate_recovery_steps(
        self,
        errors: List[ErrorDetails],
        context: Dict[str, Any] = None
    ) -> List[str]:
        """Generate recovery steps for multiple errors"""
        
        steps = []
        
        # Group errors by category
        by_category = {}
        for error in errors:
            category = error.category.value
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(error)
        
        # Generate steps for each category
        for category, category_errors in by_category.items():
            steps.append(f"Address {len(category_errors)} {category} issue(s)")
            
            # Add specific steps for critical errors
            critical_errors = [e for e in category_errors if e.severity == ErrorSeverity.CRITICAL]
            for error in critical_errors[:2]:  # Top 2 critical errors
                steps.extend(error.suggestions[:2])  # Top 2 suggestions
        
        return steps
    
    def _generate_system_recommendations(self, analytics: Dict[str, Any]) -> List[str]:
        """Generate system improvement recommendations"""
        
        recommendations = []
        
        if analytics["system_health"] == "critical":
            recommendations.append("Immediate attention required - critical errors detected")
        
        if analytics["error_by_severity"].get("error", 0) > 10:
            recommendations.append("Consider reviewing common error patterns")
        
        if analytics["total_errors"] > 50:
            recommendations.append("System may benefit from user training or process improvements")
        
        return recommendations


def create_error_handler(log_file: str = "housebrain_errors.log") -> EnhancedErrorHandler:
    """Create enhanced error handler instance"""
    
    return EnhancedErrorHandler(log_file)


if __name__ == "__main__":
    # Test enhanced error handling
    error_handler = create_error_handler()
    
    print("🛡️ Enhanced Error Handling Test")
    print("=" * 50)
    
    # Test error handling
    try:
        raise FileNotFoundError("Test file not found")
    except Exception as e:
        error_details = error_handler.handle_error(
            e,
            ErrorCategory.FILE_IO,
            {"file_path": "/test/path.json"},
            "loading test file"
        )
        print(f"Error handled: {error_details.user_message}")
        print(f"Suggestions: {error_details.suggestions[:2]}")
    
    # Test validation
    test_data = {"geometry": {"spaces": [{"area": 500000}]}}  # Too small
    validation_result = error_handler.validate_input(test_data, "house_plan")
    
    print(f"\nValidation result: {validation_result.get_error_summary()}")
    
    # Test analytics
    analytics = error_handler.get_error_analytics()
    print(f"System health: {analytics['system_health']}")
    print(f"Total errors tracked: {analytics['total_errors']}")
    
    print("\n✅ Enhanced Error Handling initialized successfully!")