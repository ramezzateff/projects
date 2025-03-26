#!/usr/bin/python3
import numpy as np
import h5py
from gnuradio import digital, blocks, gr, analog

class QAM_Generator(gr.top_block):
    def __init__(self, samp_rate=32000, num_samples=100000, noise_level=0.1):
        gr.top_block.__init__(self)

        # Configuration
        self.samp_rate = samp_rate
        self.num_samples = num_samples

        # Define 16-QAM constellation (16 symbols)
        qam_constellation = digital.constellation_16qam().base()

        # Random data generator (values from 0 to 15 for 16-QAM)
        self.random_source = blocks.vector_source_b(
            np.random.randint(0, 16, self.num_samples).tolist(), 
            False  # Not repeating
        )

        # Map digital data to 16-QAM symbols
        self.symbol_mapper = digital.chunks_to_symbols_bc(qam_constellation.points())

        # AWGN noise generator
        self.noise_source = analog.noise_source_c(analog.GR_GAUSSIAN, noise_level)

        # Signal + noise adder
        self.adder = blocks.add_cc()

        # Sink for the output data
        self.sink = blocks.vector_sink_c()

        # Debugging: Print when connections are made
        print("✅ Connecting blocks...")

        # Connect blocks
        self.connect(self.random_source, self.symbol_mapper)
        self.connect(self.symbol_mapper, (self.adder, 0))
        self.connect(self.noise_source, (self.adder, 1))
        self.connect(self.adder, self.sink)

    def run_simulation(self):
        """ Runs the FlowGraph and ensures proper execution """
        try:
            print("✅ Running FlowGraph...")
            self.start()
            self.wait()  # Wait until the flowgraph finishes
            print("✅ Execution finished.")
        except Exception as e:
            print(f"❌ Error during execution: {e}")

    def get_data(self):
        """ Retrieves data from the sink """
        if not hasattr(self, 'sink') or self.sink is None:
            print("❌ ERROR: sink is not defined!")
            return np.array([])

        # Debugging: Print number of items in sink
        data = np.array(self.sink.data(), dtype=np.complex64)
        print(f"⚙ Data length in sink: {len(data)}")
        if len(data) == 0:
            print("⚠ No data recorded! Check the flow settings.")
        return data

# Run the simulation
if __name__ == "__main__":
    qam_gen = QAM_Generator()
    qam_gen.run_simulation()

    # Retrieve data
    data = qam_gen.get_data()

    if len(data) > 0:
        print(f"✅ Recorded samples: {len(data)}")
        # Save data to a file
        with h5py.File("qam_data.h5", "w") as f:
            f.create_dataset("qam_signal", data=data)
        print("✅ Data saved to qam_data.h5")
    else:
        print("⚠ No data recorded, check the flow settings.")
