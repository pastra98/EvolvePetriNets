import sys
import json
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QCheckBox, QPushButton, QFileDialog, 
                             QScrollArea, QComboBox, QSpinBox, QTextEdit, QInputDialog, QMessageBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap
from itertools import product
import graphviz

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Genetic Algorithm Configuration")
        self.setGeometry(100, 100, 800, 600)

        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Config section
        config_widget = QWidget()
        config_layout = QVBoxLayout()
        config_widget.setLayout(config_layout)
        
        self.logpath = QLineEdit()
        self.logpath.setText("pm_data/running_example.xes")
        config_layout.addWidget(QLabel("Log Path:"))
        config_layout.addWidget(self.logpath)
        
        self.serialize_pop = QCheckBox("Serialize Population")
        config_layout.addWidget(self.serialize_pop)
        
        self.time_execution = QCheckBox("Time Execution")
        config_layout.addWidget(self.time_execution)
        
        self.stop_after = QSpinBox()
        self.stop_after.setRange(1, 10000)
        self.stop_after.setValue(500)
        self.stop_after.valueChanged.connect(self.update_info_box)
        config_layout.addWidget(QLabel("Number of Generations:"))
        config_layout.addWidget(self.stop_after)
        
        self.setup_runs = QSpinBox()
        self.setup_runs.setRange(1, 1000)
        self.setup_runs.setValue(4)
        self.setup_runs.valueChanged.connect(self.update_info_box)
        config_layout.addWidget(QLabel("Number of Setup Runs:"))
        config_layout.addWidget(self.setup_runs)
        
        self.config_description = QTextEdit()
        config_layout.addWidget(QLabel("Config Description:"))
        config_layout.addWidget(self.config_description)
        
        self.info_box = QLabel()
        config_layout.addWidget(self.info_box)
        
        self.save_config_button = QPushButton("Save Config")
        self.save_config_button.clicked.connect(self.save_config)
        config_layout.addWidget(self.save_config_button)
        
        # Parameters section
        params_widget = QWidget()
        params_layout = QVBoxLayout()
        params_widget.setLayout(params_layout)
        
        self.load_params_button = QPushButton("Load Params")
        self.load_params_button.clicked.connect(self.load_params)
        params_layout.addWidget(self.load_params_button)
        
        self.base_params_info = QLabel()
        params_layout.addWidget(self.base_params_info)
        
        self.params_scroll = QScrollArea()
        self.params_scroll.setWidgetResizable(True)
        self.params_content = QWidget()
        self.params_content_layout = QVBoxLayout()
        self.params_content.setLayout(self.params_content_layout)
        self.params_scroll.setWidget(self.params_content)
        params_layout.addWidget(self.params_scroll)
        
        param_buttons_layout = QHBoxLayout()
        self.add_param_button = QPushButton("Add New Parameter")
        self.add_param_button.clicked.connect(self.add_parameter)
        param_buttons_layout.addWidget(self.add_param_button)
        
        self.remove_param_button = QPushButton("Remove Parameter")
        self.remove_param_button.clicked.connect(self.remove_parameter)
        param_buttons_layout.addWidget(self.remove_param_button)
        
        params_layout.addLayout(param_buttons_layout)

        # Tree visualization section
        tree_widget = QWidget()
        tree_layout = QVBoxLayout()
        tree_widget.setLayout(tree_layout)
        
        self.tree_label = QLabel()
        tree_layout.addWidget(self.tree_label)
        
        self.checkbox_scroll = QScrollArea()
        self.checkbox_scroll.setWidgetResizable(True)
        self.checkbox_content = QWidget()
        self.checkbox_layout = QVBoxLayout()
        self.checkbox_content.setLayout(self.checkbox_layout)
        self.checkbox_scroll.setWidget(self.checkbox_content)
        tree_layout.addWidget(self.checkbox_scroll)

        
        # Add sections to main layout
        main_layout.addWidget(config_widget)
        main_layout.addWidget(params_widget)
        main_layout.addWidget(tree_widget)

        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        self.base_params = {}
        self.param_changes = {}
        
        # Load default params
        self.load_params("params/default_params.json")
        self.update_info_box()

        
    def load_params(self, file_path=None):
        if file_path is None:
            file_name, _ = QFileDialog.getOpenFileName(self, "Open Params File", "", "JSON Files (*.json)")
        else:
            file_name = file_path
        
        if file_name:
            with open(file_name, 'r') as f:
                self.base_params = json.load(f)
            self.update_info_box()
            self.base_params_info.setText(f"Currently loaded: {file_name}")
    

    def add_parameter(self):
        param_widget = QWidget()
        param_layout = QHBoxLayout()
        
        param_key = QComboBox()
        param_key.addItems(self.get_nested_keys(self.base_params))
        param_layout.addWidget(param_key)
        
        param_values = QLineEdit()
        param_values.editingFinished.connect(self.update_tree)
        param_values.editingFinished.connect(self.update_info_box)
        param_layout.addWidget(param_values)
        
        param_widget.setLayout(param_layout)
        self.params_content_layout.addWidget(param_widget)
        self.update_tree()


    def remove_parameter(self):
        if self.params_content_layout.count() > 0:
            item = self.params_content_layout.takeAt(self.params_content_layout.count() - 1)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            self.update_tree()


    def update_tree(self):
        param_values = self.get_param_values()
        if not param_values:
            return

        dot = graphviz.Digraph(comment='Parameter Tree')
        dot.attr(rankdir='TB')  # Top to Bottom layout

        def create_node_label(params):
            return '\n'.join([f"{k}: {v}" for k, v in params.items()])

        def add_nodes(current_params, remaining_params, parent_id=None):
            if not remaining_params:
                setup_number = len(dot.body) // 2 + 1  # Approximate setup number
                node_id = f"setup_{setup_number}"
                dot.node(node_id, f"Setup {setup_number}\n{create_node_label(current_params)}")
                if parent_id:
                    dot.edge(parent_id, node_id)
                return

            current_key, current_values = next(iter(remaining_params.items()))
            new_remaining = dict(list(remaining_params.items())[1:])

            for value in current_values:
                new_params = {**current_params, current_key: value}
                node_id = f"node_{'_'.join(map(str, new_params.values()))}"
                dot.node(node_id, create_node_label(new_params))

                if parent_id:
                    dot.edge(parent_id, node_id)

                add_nodes(new_params, new_remaining, node_id)

        # Add root node
        root_id = "root"
        dot.node(root_id, "base params")

        # Start the recursion with the root node as parent
        add_nodes({}, param_values, root_id)

        # Render the tree
        dot.render('tree', format='png', cleanup=True)
        
        # Display the tree
        pixmap = QPixmap('tree.png')
        self.tree_label.setPixmap(pixmap.scaled(600, 400, Qt.AspectRatioMode.KeepAspectRatio))

        # Update checkboxes
        num_setups = len(list(product(*param_values.values())))
        self.update_checkboxes(num_setups)


    def get_param_values(self):
        param_values = {}
        for i in range(self.params_content_layout.count()):
            widget = self.params_content_layout.itemAt(i).widget()
            key = widget.layout().itemAt(0).widget().currentText()
            values = widget.layout().itemAt(1).widget().text().split(';')
            param_values[key] = values
        return param_values


    def update_checkboxes(self, num_setups):
        # Clear existing checkboxes
        for i in reversed(range(self.checkbox_layout.count())): 
            self.checkbox_layout.itemAt(i).widget().setParent(None)

        # Add new checkboxes
        for i in range(num_setups):
            checkbox = QCheckBox(f'Setup {i+1}')
            checkbox.setChecked(True)
            self.checkbox_layout.addWidget(checkbox)


    def get_nested_keys(self, d, prefix=''):
        keys = []
        for k, v in d.items():
            new_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                keys.extend(self.get_nested_keys(v, new_key))
            else:
                keys.append(new_key)
        return keys
    
    def save_config(self):
        name, ok = QInputDialog.getText(self, "Save Config", "Enter config name:")
        if ok and name:
            config = {
                "name": name,
                "setups": []
            }
            
            param_combinations = self.generate_param_combinations()

            config_dir = f"configs/{name}"
            os.makedirs(config_dir, exist_ok=True)
            
            for i, params in enumerate(param_combinations):
                setup = {
                    "setupname": f"setup_{i+1}",
                    "parampath": f"{config_dir}/setup_{i+1}.json",
                    "logpath": self.logpath.text(),
                    "ga_kwargs": {
                        "is_pop_serialized": self.serialize_pop.isChecked(),
                        "is_timed": self.time_execution.isChecked()
                    },
                    "stop_cond": {
                        "var": "gen",
                        "val": self.stop_after.value()
                    },
                    "n_runs": self.setup_runs.value(),
                    "send_gen_info_to_console": False,
                    "is_profiled": False
                }
                config["setups"].append(setup)
            
            with open(f"{config_dir}/{name}.json", 'w') as f:
                json.dump(config, f, indent=4)

            for i, params in enumerate(param_combinations):
                with open(f"{config_dir}/setup_{i+1}.json", 'w') as f:
                    json.dump(params, f, indent=4)
            
            with open(f"{config_dir}/param_changes.txt", 'w') as f:
                f.write(f"Config Description:\n{self.config_description.toPlainText()}\n\n")
                for i, changes in enumerate(self.param_changes):
                    f.write(f"Setup {i+1}:\n")
                    for key, value in changes.items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")
            
            self.show_notification("Config saved successfully!")
    
    def generate_param_combinations(self):
        param_values = {}
        for i in range(self.params_content_layout.count()):
            widget = self.params_content_layout.itemAt(i).widget()
            key = widget.layout().itemAt(0).widget().currentText()
            values = widget.layout().itemAt(1).widget().text().split(';')
            param_values[key] = values
        
        keys, value_lists = zip(*param_values.items()) if param_values else ([], [])
        combinations = list(product(*value_lists)) if value_lists else [[]]
        
        param_combinations = []
        self.param_changes = []
        
        for combo in combinations:
            new_params = self.base_params.copy()
            changes = {}
            for key, value in zip(keys, combo):
                nested_set(new_params, key.split('.'), parse_value(value))
                changes[key] = value
            param_combinations.append(new_params)
            self.param_changes.append(changes)
        
        return param_combinations
    

    def update_info_box(self):
        num_setups = max(1, len(self.generate_param_combinations()))

        num_runs = num_setups * self.setup_runs.value()
        total_generations = num_runs * self.stop_after.value()
        
        info_text = f"Number of setups: {num_setups}\n"
        info_text += f"Total runs: {num_runs}\n"
        info_text += f"Total generations: {total_generations}"
        
        self.info_box.setText(info_text)
    

    def show_notification(self, message):
        notification = QMessageBox(self)
        notification.setText(message)
        notification.setWindowTitle("Notification")
        notification.setStandardButtons(QMessageBox.StandardButton.NoButton)
        notification.show()
        QTimer.singleShot(500, notification.accept)


def nested_set(dic, keys, value):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value


def parse_value(value):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())