"""
Unit tests for input validation and location error handling in predict.py.
"""
import unittest
from src.model.predict import (
    validate_cap,
    validate_machine_type,
    validate_comune,
    validate_address,
    validate_store_name,
    RVMNewLocationPredictor
)

class TestInputValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = RVMNewLocationPredictor()

    def test_validate_cap_valid(self):
        self.assertEqual(validate_cap("50019"), "50019")
        self.assertEqual(validate_cap("20121"), "20121")
        self.assertEqual(validate_cap("56028"), "56028")
        self.assertEqual(validate_cap("00100"), "00100")
        # Postal codes with less than 5 digits (e.g. 7100 for Sassari, 185 for Rome, 123)
        self.assertEqual(validate_cap("7100"), "07100")
        self.assertEqual(validate_cap("185"), "00185")
        self.assertEqual(validate_cap("123"), "00123")
        self.assertEqual(validate_cap(7100), "07100")

    def test_validate_cap_invalid(self):
        # Text string like the one entered by the user
        with self.assertRaises(ValueError) as ctx:
            validate_cap("Piazza Falcone e Borsellino 3")
        self.assertIn("Location not found: Invalid CAP", str(ctx.exception))

        # Too long (exceeds 5 digits)
        with self.assertRaises(ValueError):
            validate_cap("123456")

        # Invalid prefix (99 does not exist in Italy)
        with self.assertRaises(ValueError) as ctx:
            validate_cap("99999")
        self.assertIn("Invalid CAP prefix", str(ctx.exception))

    def test_validate_machine_type_valid(self):
        self.assertEqual(validate_machine_type("DAINT_G"), "DAINT_G")
        self.assertEqual(validate_machine_type("Ripet_C-front"), "Ripet_C-front")
        self.assertEqual(validate_machine_type("DAINT_A-bis"), "DAINT_A-bis")
        self.assertEqual(validate_machine_type("Ripet_V2"), "Ripet_V2")

    def test_validate_machine_type_invalid(self):
        # Street address entered as machine type
        with self.assertRaises(ValueError) as ctx:
            validate_machine_type("Piazza Falcone e Borsellino 3")
        self.assertIn("Invalid machine type", str(ctx.exception))

        # Bogus string
        with self.assertRaises(ValueError):
            validate_machine_type("XYZ_TEST_MACHINE_123")

    def test_validate_comune_valid(self):
        self.assertEqual(validate_comune("Milano"), "Milano")
        self.assertEqual(validate_comune("Firenze"), "Firenze")
        self.assertEqual(validate_comune("Pistoia"), "Pistoia")
        self.assertEqual(validate_comune("Sesto Fiorentino"), "Sesto Fiorentino")
        self.assertEqual(validate_comune("Cantù"), "Cantù")

    def test_validate_comune_invalid_street(self):
        # Street address entered as comune
        with self.assertRaises(ValueError) as ctx:
            validate_comune("Piazza Falcone e Borsellino 3")
        self.assertIn("appears to be a street address", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            validate_comune("Via Roma 10")
        self.assertIn("appears to be a street address", str(ctx.exception))

    def test_validate_comune_identical_to_address(self):
        with self.assertRaises(ValueError) as ctx:
            validate_comune("Corso Garibaldi", address="Corso Garibaldi")
        self.assertIn("cannot be identical to the street address", str(ctx.exception))

    def test_validate_comune_unrecognized(self):
        with self.assertRaises(ValueError) as ctx:
            validate_comune("AtlantideCittaFantasma")
        self.assertIn("not recognized as a valid Italian Comune", str(ctx.exception))

    def test_validate_address_and_store_name(self):
        self.assertEqual(validate_address("Via Roma, 10"), "Via Roma, 10")
        with self.assertRaises(ValueError):
            validate_address("")
        with self.assertRaises(ValueError):
            validate_address("12")

        self.assertEqual(validate_store_name("Coop.fi"), "Coop.fi")
        with self.assertRaises(ValueError):
            validate_store_name("")

    def test_resolve_location_regional_conflict(self):
        # CAP 00100 is Rome (Lazio), but Comune is Milano (Lombardia)
        with self.assertRaises(ValueError) as ctx:
            self.predictor.resolve_location(address="Via Roma 1", cap="00100", comune="Milano")
        self.assertIn("Regional conflict between CAP", str(ctx.exception))

    def test_resolve_location_unresolvable_throws_error(self):
        # Non-existent Italian CAP / location
        with self.assertRaises(ValueError) as ctx:
            self.predictor.resolve_location(address="Via Inesistente 999", cap="99999", comune="Milano")
        self.assertIn("Location not found", str(ctx.exception))

    def test_predict_new_location_rejects_user_screenshot_data(self):
        # Exact data from the user screenshot: Piazza Falcone e Borsellino 3 in all fields
        with self.assertRaises(ValueError) as ctx:
            self.predictor.predict_new_location(
                store_name="Piazza Falcone e Borsellino 3",
                address="Piazza Falcone e Borsellino 3",
                cap="Piazza Falcone e Borsellino 3",
                machine_type="Piazza Falcone e Borsellino 3",
                comune="Piazza Falcone e Borsellino 3"
            )
        self.assertIn("Location not found", str(ctx.exception))

    def test_predict_new_location_valid_succeeds(self):
        # Legitimate verified store
        res = self.predictor.predict_new_location(
            store_name="Coop.fi",
            address="Viale Pratese, 33/35",
            cap="50019",
            machine_type="Ripet_C-front",
            comune="Sesto Fiorentino"
        )
        self.assertIn("predictions", res)
        self.assertIn("realized_forecast", res)
        self.assertEqual(res["location_attributes"]["resolved_comune"], "Sesto Fiorentino")

if __name__ == "__main__":
    unittest.main()
