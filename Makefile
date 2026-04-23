.PHONY: setup font synth sample train eval demo clean

setup:
	uv venv
	uv pip install -e ".[train,serve,demo,dev]"

font:
	@test -f assets/fonts/SarunsThangLuang.ttf \
		|| (echo "Missing assets/fonts/SarunsThangLuang.ttf — see assets/fonts/README.md" && exit 1)
	@echo "Font OK."

synth: font
	uv run python -m thai_plate_synth.render --out data/synth_v1 --count 1000

sample: font
	uv run python -m thai_plate_synth.render --out experiments/figures/samples --count 10 --seed 0

train:
	uv run python -m thai_plate_synth.train --data data/synth_v1 --epochs 50

eval:
	uv run python -m thai_plate_synth.eval --weights runs/detect/train/weights/best.pt

demo:
	uv run streamlit run app/streamlit_app.py

clean:
	rm -rf data/synth_v* runs/ experiments/runs/
