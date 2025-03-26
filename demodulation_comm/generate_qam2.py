#!/usr/bin/python3
import numpy as np
import h5py
from gnuradio import digital, blocks, gr, analog

class QAM_Generator(gr.top_block):
    def __init__(self, samp_rate=32000, num_samples=100000, noise_level=0.1):
        super(QAM_Generator, self).__init__()

        # إعداد القيم
        self.samp_rate = samp_rate
        self.num_samples = num_samples

        # تعريف كوكبة 16-QAM
        qam_constellation = digital.constellation_16qam().base()

        # توليد بيانات عشوائية بين 0 و 15
        self.random_source = blocks.vector_source_b(
            np.random.randint(0, 16, self.num_samples).tolist(), False
        )

        # تحويل البيانات الرقمية إلى رموز 16-QAM
        self.symbol_mapper = digital.chunks_to_symbols_bc(qam_constellation.points())

        # توليد الضوضاء
        self.noise_source = analog.noise_source_c(analog.GR_GAUSSIAN, noise_level)

        # إضافة الإشارة مع الضوضاء
        self.adder = blocks.add_cc()

        # مخرج البيانات
        self.sink = blocks.vector_sink_c()

        # توصيل المكونات
        self.connect(self.random_source, self.symbol_mapper)
        self.connect(self.symbol_mapper, (self.adder, 0))
        self.connect(self.noise_source, (self.adder, 1))
        self.connect(self.adder, self.sink)

    def run_simulation(self):
        """تشغيل المخطط"""
        self.start()
        self.wait()

    def get_data(self):
        """استخراج البيانات"""
        return np.array(self.sink.data(), dtype=np.complex64)

# تشغيل المحاكي
if __name__ == "__main__":
    qam_gen = QAM_Generator()
    qam_gen.run_simulation()

    # حفظ البيانات
    data = qam_gen.get_data()
    if len(data) > 0:
        with h5py.File("qam_data.h5", "w") as f:
            f.create_dataset("qam_signal", data=data)
        print("✅ Data saved to qam_data.h5")
    else:
        print("❌ No data recorded, check settings.")
