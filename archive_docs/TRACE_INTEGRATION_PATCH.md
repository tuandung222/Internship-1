# Trace Integration Patch for test_real_pipeline.py

## Changes needed in test_real_pipeline.py

### 1. Update imports (around line 46-53)
Already done - imports added.

### 2. Update argument parser (around line 530-550)
Already done - tracing flags added.

### 3. Initialize loggers (around line 555-567)
Already done - loggers initialized.

### 4. Update VLMClientFactory.create_from_config call (around line 610)
Change:
```python
client = VLMClientFactory.create_from_config(config)
```
To:
```python
client = VLMClientFactory.create_from_config(
    config,
    image_logger=image_logger,
    output_tracer=output_tracer,
)
```

### 5. Update CoRGIPipeline initialization (around line 611)
Change:
```python
pipeline = CoRGIPipeline(vlm_client=client)
```
To:
```python
pipeline = CoRGIPipeline(
    vlm_client=client,
    image_logger=image_logger,
    output_tracer=output_tracer,
)
```

### 6. Add trace report generation (after line 695, before final summary)
Add this code block:
```python
    # Generate trace report if tracing enabled
    if enable_tracing and image_logger and output_tracer:
        console.print("\n[bold cyan]Generating Trace Report...[/bold cyan]")
        try:
            # Create pipeline trace
            pipeline_trace = PipelineTrace(
                pipeline_id=f"pipeline_{timestamp}",
                question=question,
                config={
                    "reasoning": {
                        "model_id": config.reasoning.model.model_id,
                        "model_type": config.reasoning.model.model_type,
                        "max_steps": config.reasoning.max_steps,
                    },
                    "grounding": {
                        "model_id": config.grounding.model.model_id,
                        "model_type": config.grounding.model.model_type,
                        "max_regions": config.grounding.max_regions,
                    },
                    "captioning": {
                        "model_id": config.captioning.model.model_id,
                        "model_type": config.captioning.model.model_type,
                    },
                    "synthesis": {
                        "model_id": config.synthesis.model.model_id,
                        "model_type": config.synthesis.model.model_type,
                    },
                },
                start_timestamp=datetime.utcnow().isoformat() + "Z",
                end_timestamp=datetime.utcnow().isoformat() + "Z",
                total_duration_ms=result.total_duration_ms,
                original_image_path=str(args.output_dir / "images" / "original" / "input_image.png"),
                final_result={
                    "answer": result.answer,
                    "steps_count": len(result.steps),
                    "evidence_count": len(result.evidence),
                    "key_evidence_count": len(result.key_evidence),
                },
            )
            
            # Save image metadata
            image_logger.save_metadata_summary()
            
            # Save trace summary
            output_tracer.save_summary()
            
            # Generate HTML report
            report_generator = HTMLReportGenerator(trace_dir)
            html_path = report_generator.generate_report(
                pipeline_trace,
                image_logger,
                output_tracer,
            )
            
            # Save pipeline trace JSON
            trace_json_path = trace_dir / f"pipeline_trace_{timestamp}.json"
            pipeline_trace.save_json(trace_json_path)
            
            console.print(f"[green]✓[/green] Trace report generated: [cyan]{html_path}[/cyan]")
            console.print(f"[green]✓[/green] Pipeline trace saved: [cyan]{trace_json_path}[/cyan]")
            
        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] Failed to generate trace report: {e}")
            import traceback
            console.print(traceback.format_exc())
```

### 7. Update final summary (around line 700)
Change:
```python
    console.print(f"\nResults saved to: [cyan]{args.output_dir}[/cyan]")
```
To:
```python
    console.print(f"\nResults saved to: [cyan]{args.output_dir}[/cyan]")
    if enable_tracing:
        console.print(f"Trace data saved to: [cyan]{trace_dir}[/cyan]")
```

